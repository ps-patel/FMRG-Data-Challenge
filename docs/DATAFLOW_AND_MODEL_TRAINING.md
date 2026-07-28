# Data Flow & Model Training — Complete Methodology

NSF FMRG Data Challenge · DED laser-track local geometry prediction
Companion to `FINAL_REPORT.md` (results) and `code/INDEX.md` (script map). This document explains **how data moves through the pipeline** and **why every modeling choice was made**.

---

## 1. End-to-end data flow

```
RAW DATA (Zenodo, 667 MB)
│
├── thermal/Thermal_{8,10,14,21}.mat          93–105 MB each, MATLAB v5
│     (929–1012 frames × 400 × 400 px, 50 fps, 14 µm/px, camera counts)
│
├── height_maps/Heightmap_{id}.ASC            ~280 MB each, Wyko ASCII
│     (480 × ~20,000 grid, 3.982 µm/px, z in nm, 37–56% NaN, x=100 on RIGHT)
│
└── sem/SEM_{id}/PlainImages/*.tif            13–14 tiles/track, 768×1024 uint8
      (~6.41 mm/tile, tile 01 at the 100 mm side, x=100 on LEFT)

────────────────────────── STAGE A: GROUND TRUTH (height maps → labels) ──────────────────────────
 A1  load_wyko_asc()          nm→mm, reverse x to physical order, crop to 20–100 mm
 A2  robust_plane_detrend()   iterative least-squares plane fit, track band excluded,
                              5–95% residual trimming (3 rounds)
 A3  Melt-zone score map      S = box_mean(validity) × exp(−box_mean(|dz/dy|)/2)
                              KEY INSIGHT: no-powder remelt tracks are nearly flush with the
                              substrate (height thresholding fails: 27/400 bins) but optically
                              SMOOTH — 4–15% NaN inside the track vs 60–80% on rough substrate.
                              The melt boundary is where smooth+valid ends.
 A4  Per-column boundary      threshold S (85th/30th percentile midpoint), largest run with
                              NaN-bridging ≤12 px, minimum width 0.15 mm
 A5  Bin to 0.2 mm grid       (matches 1 thermal frame of laser travel)
                              → width(x), y_left(x), y_right(x), edge_rough(x), valid_frac(x)
 A6  Extraction ensemble      re-run A4 over 6 hyperparameter variants
                              → ensemble-median label + per-bin spread w_spread (13 µm median)
 A7  Z-axis descriptors       z_peak, z_mean, z_valley, z_range per bin vs substrate plane

────────────────────────── STAGE B: THERMAL FEATURES (.mat → 69 numbers/bin) ─────────────────────
 B1  loadmat_v5()             custom pure-numpy MAT reader (zlib-compressed elements)
 B2  Laser-window detection   99.5th-percentile brightness per frame; robust threshold
                              (max of range-based and 8×MAD rules); largest contiguous run.
                              28–31% of frames are pre/post-laser junk → dropped.
                              Detected windows match dataset-paper Table 2 exactly (all 4 tracks).
 B3  Final 400 frames         before laser-off = 20–100 mm at 0.2 mm/frame;
                              frame i stamped x = 100 − (frames_before_off − 0.5)·0.2
 B4  23 descriptors/frame     areas above 1400/1600/1800/2000/2200 counts; peak & p99.9
                              intensity; energy proxy; centroid (x,y); pool length/width/
                              eccentricity/asymmetry (2nd moments); cooling-tail length & decay
 B5  Rolling stats            ±1 mm (±5 bin) mean and std per descriptor → ~69 features/bin
 B6  Lag alignment            all thermal features shifted +14 bins (+2.8 mm), chosen by
                              maximizing within-track feature↔width correlation on TRAINING
                              tracks only (oracle lags later confirmed +14…+17 on all tracks)

────────────────────────── STAGE C: SEM FEATURES (tiles → 7 numbers/bin) ─────────────────────────
 C1  Stitch mosaic            tiles 01→N left-to-right, 5% overlap (organizer spec),
                              no per-tile flips
 C2  fliplr(whole mosaic)     → same left-to-right direction as height map (x=100 on right)
 C3  Track masking            track row = min row-gradient (smooth melt band);
                              ±0.75 mm EXCLUDED (leakage rule; ±0.45 mm proved too narrow
                              on the 400/350 W tracks — caught by visual audit, fig7/fig11)
 C4  Flank-only texture       per 0.2 mm slice: mean, std, gradient energy, local variance,
                              low/high-frequency FFT power, entropy → 7 features/bin

────────────────────────── STAGE D: MODELING TABLE ───────────────────────────────────────────────
 D1  Join on the 0.2 mm grid  400 bins × 4 tracks = 1,600 rows
     columns: 69 thermal + 7 SEM + power  |  labels: width/boundaries/z + quality columns
 D2  Sample weights           w = clip(valid_frac, 0.05) / (1 + w_spread/30)
                              → noisy label bins matter less in training

────────────────────────── STAGE E: TRAINING & EVALUATION ────────────────────────────────────────
 E1  Splits    LOTO CV over tracks 8/10/14 (train 2, validate 1, rotate)
               Track 21 (200 W) = untouched holdout, scored ONCE
               + blocked within-track CV (16 mm blocks, 2 mm guard) for local skill
 E2  Models    point GBM (L2) + 5 quantile GBMs (pinball, q=.05,.25,.5,.75,.95) for width;
               point GBM for centerline; boundaries = c ∓ w/2
 E3  Calibration  quantiles recentered on the weighted point model; conformal multiplier
               s = 2.60 chosen on train folds: minimize CRPS subject to ≥88% coverage
 E4  Outputs   w(x), y_left(x), y_right(x), c(x), z_range(x) — each with 50%/90% bands
```

---

## 2. Why each model was chosen (and rejected)

### The constraint that drives everything: n = 4 tracks

Only 4 tracks exist, each at a unique laser power (400/350/300/200 W). Power is therefore
**perfectly aliased with track identity**, every evaluation fold is a power extrapolation or
interpolation, and any model with enough capacity to memorize track-level patterns will do so.
Model choice is dominated by *regularization and extrapolation behavior*, not raw capacity.

### Considered, chosen, rejected

| Model | Decision | Reasoning | Evidence (T21 MAE) |
|---|---|---|---|
| Naive mean | baseline | The bar every model must beat; also exposes how much skill is track-level | 257 µm |
| **Ridge (linear)** | rejected | Linear extrapolation amplifies out-of-range power: predictions leave the physical range entirely | **582 µm — 2.3× worse than doing nothing** |
| **kNN** | rejected (kept as reference) | No extrapolation at all (clamps to nearest training samples) — survives but can never beat the nearest-track answer; unstable across folds (301 µm on held-T10) | 116 µm |
| **Random forest** | rejected | Same clamping as kNN with more variance at this n; no benefit over GBM in LOTO | ~GBM, noisier |
| **Histogram GBM (chosen)** | **primary model** | (1) handles NaNs & mixed scales natively; (2) depth-3 trees + subsampling = strong regularization at n≈1,100 train rows; (3) pinball loss gives the full predictive distribution the challenge scores (CRPS/calibration); (4) sample weights integrate label quality; (5) monotone-ish response to the dominant features extrapolates better than linear | **93–97 µm** |
| **Quantile GBM ensemble** | probabilistic layer | CRPS/NLL/calibration are first-class metrics; 5 quantiles ≈ predictive CDF; recentered on the weighted point model | CRPS 85 µm after calibration |
| **Conformal calibration** | required add-on | Raw quantile intervals under-cover badly under power extrapolation (cov90 = 0.54); conformal multiplier from train folds restores validity; tuned to CRPS not just coverage | cov90 0.54 → 1.00 |
| **√P physics head** | rejected by data | Width vs power is strongly nonlinear (530→559→900 µm at 300/350/400 W); a 2–3-point scaling law extrapolates catastrophically | held-14 MAE 316→421 |
| **Physics anchor (melt-pool area regression)** | rejected via honest selection | Chosen on train-LOTO it scores 119 µm alone; blending weight α tuned on train folds → α=1.0 (pure GBM). Post-hoc, a melt-pool *width* anchor would have hit 61 µm — but selecting it required peeking at the test track, so it was not used | documented for future validation |
| **Deep CNN / fusion nets** | not attempted (deliberately) | 4 tracks cannot support representation learning; blocked-CV local correlation (~0.1) shows the ceiling is label/alignment quality, not model capacity. Spectral features confirmed the missing local signal is above the camera's Nyquist limit — no architecture can recover it | — |
| **XGBoost** | validation only | Same algorithm family as the custom GBM; run to verify the from-scratch implementation. Statistical tie on holdout (93 vs 97), worse mean LOTO (214 vs 176) and worse calibrated CRPS (103 vs 85) | 93 µm |

### Why the GBM is implemented from scratch (numpy)

The analysis sandbox had no scipy/sklearn/lightgbm and its package registry was blocked.
`np_models.py` implements histogram gradient boosting directly: 32 quantile bins per feature,
depth-3 trees, exact split search via cumulative histogram sums, gradient boosting for L2 and
quantile-optimal leaf updates for pinball loss, subsampling 0.9 row/column, weighted resampling
for sample weights. The XGBoost head-to-head (§6 of the final report) confirms it matches the
industry implementation.

### Final hyperparameters (all selected on training tracks only)

| Parameter | Value | Why |
|---|---|---|
| Trees | 150 point / 100 per quantile | Loss plateaus by ~120 trees; more risks track memorization |
| Depth | 3 | 8 leaves ≈ enough for threshold physics; depth 5+ overfits folds |
| Learning rate | 0.08 | Standard for 100–200 trees |
| Bins | 32 | n≈1,100 rows — finer binning adds noise, not signal |
| Subsample / colsample | 0.9 / 0.9 | Decorrelates trees; full bagging hurt at this n |
| Thermal lag | +14 bins | Max within-track corr on train tracks |
| Sample weight | valid_frac / (1 + w_spread/30) | Down-weights NaN-heavy and extraction-sensitive bins |
| Conformal s | 2.60 | Min train-fold CRPS s.t. coverage ≥ 88% |
| Quantiles | .05 .25 .50 .75 .95 | CRPS approximation + 50%/90% intervals |

---

## 3. Leakage controls baked into the flow

1. Height maps appear **only** in Stage A (labels) — never as features.
2. SEM track band ±0.75 mm excluded from all features; track pixels used only to *locate* the mask.
3. GBM bin edges, scalers, lag, anchor selection, blend weight, conformal multiplier: all fit on
   **training folds only**.
4. Track 21 was touched exactly once per final configuration, after all choices were frozen.
5. Rolling features span ±1 mm < the 2 mm guard band of blocked CV.
6. Label-side columns (width, boundaries, quality flags) are dropped from the feature matrix.

## 4. What the trained system outputs

For any new track (thermal video + substrate SEM + power setting):

| Output | Meaning | Validated accuracy (unseen 200 W track) |
|---|---|---|
| w(x) | local track width every 0.2 mm | 93–97 µm MAE, CRPS 85 µm |
| y_left(x), y_right(x) | track boundary functions (capture shift & asymmetry) | 107 / 67 µm MAE |
| c(x) | centerline displacement | 74 µm MAE |
| z_range(x) | resolidification ripple amplitude (z-texture) | 1.7 µm MAE (43% < baseline) |
| 50% / 90% bands | calibrated uncertainty on all of the above | coverage 0.95 / 1.00 |

Known limits, by design and documented: net z-elevation (±10 µm, non-monotonic in power) is not
predictable from 3 training powers; the exact point-to-point width ripple is bounded by label
noise (55–77% of local variance) and the 25 Hz camera Nyquist limit vs 33–100 Hz ripple
formation — it is expressed as the width of the calibrated bands, not as a wiggle in the
point prediction.

---

*Scripts for every stage: see `code/INDEX.md`. Results and figures: `FINAL_REPORT.md` / `.docx`.*
