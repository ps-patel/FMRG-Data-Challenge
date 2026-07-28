# NSF FMRG Data Challenge — Probabilistic Local Geometry Prediction in DED Laser Tracks

End-to-end, fully reproducible pipeline: raw multimodal data (thermal melt-pool video,
white-light profilometry, SEM) → ground-truth extraction → feature engineering →
probabilistic gradient-boosted model → calibrated predictions of local track geometry.

**Headline result (Track 21 holdout, 200 W, never seen in training):**
width MAE 93–97 µm (62% below baseline) · boundaries 67/107 µm · centerline 74 µm ·
z-ripple amplitude 1.7 µm · CRPS 85 µm with 90%-interval coverage 1.00.

Method details: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) ·
[`docs/DATAFLOW_AND_MODEL_TRAINING.md`](docs/DATAFLOW_AND_MODEL_TRAINING.md) ·
full results report in `docs/FINAL_REPORT.md`.

## Quick start

Works on Windows, macOS, and Linux. Requires **Python ≥ 3.10** and ~4 GB free disk.

```bash
# 1. install dependencies (only 4, all pure-wheel installs)
pip install -r requirements.txt

# 2. download the data (667 MB) from Zenodo:
#    https://doi.org/10.5281/zenodo.21285367
#    files: thermal.zip, height_maps.zip, sem.zip
#    Extract all three into ./data so you have:
#      data/thermal/Thermal_{8,10,14,21}.mat
#      data/height_maps/Heightmap_{8,10,14,21}.ASC
#      data/sem/SEM_{8,10,14,21}/PlainImages/*.tif

# 3. run everything (about 10-20 minutes total)
python run_pipeline.py
```

Outputs land in `cache/` (intermediate arrays + `final_results.json`) and `figures/`.

Resume or re-run pieces: `python run_pipeline.py --from 4`, `--only 7`.
Data in a different location? Set `FMRG_DATA=/path/to/data` (see `src/fmrg_paths.py`).

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `src/s0_ingest.py` | Parse MATLAB v5 thermal videos (custom reader, no scipy needed); detect laser-on window, keep final 400 frames (= 20–100 mm at 0.2 mm/frame). Parse Wyko ASCII height maps (nm→mm, x reversed to physical order). |
| 1 | `src/s1_labels.py` | **Ground truth**: melt-zone score = validity × smoothness (the no-powder remelt track is optically smooth: 4–15% NaN inside vs 60–80% outside); per-column boundary extraction → width/boundaries per 0.2 mm bin. |
| 2 | `src/s2_thermal_features.py` | 23 melt-pool descriptors per frame (areas above 1400–2200 counts, peak T, centroid, shape moments, cooling tail). |
| 3 | `src/s3_sem_features.py` | SEM mosaics (organizer spec: 5% overlap stitch, whole-mosaic fliplr); track ±0.75 mm masked (leakage rule); 7 flank-texture features per bin. |
| 4 | `src/s4_dataset.py` | Join everything on the 0.2 mm grid: 1,600 rows × ~80 features, with ±1 mm rolling stats. |
| 5 | `src/s5_label_ensemble.py` | Re-extract labels over 6 hyperparameter variants → ensemble-median label + per-bin uncertainty (used as sample weights and reported label error). |
| 6 | `src/s6_zprofile.py` | Z-axis descriptors (peak/mean/valley/ripple-amplitude vs substrate plane). |
| 7 | `src/s7_final_model.py` | **Final model**: +14-bin lag alignment, quality-weighted histogram-GBM (pure numpy, in `src/np_models.py`) — point + 5 quantile models; CRPS-tuned conformal calibration on training folds; Track-21 scorecard. |
| 8 | `src/s8_xgboost_compare.py` | *Optional*: XGBoost head-to-head on identical folds (`pip install xgboost` first). |
| 9 | `src/s9_figures.py` | Score maps, width-vs-power, final calibrated prediction figures. |

`src/extras/` holds research scripts from the analysis (alignment/spectral/registration
experiments — several deliberately negative results, documented in the reports). They may
need small path adjustments; they are kept for transparency, not as part of the pipeline.

## Why these modeling choices (short version)

Only 4 tracks exist and laser power is perfectly aliased with track identity, so every
evaluation fold is a power extrapolation/interpolation. Histogram gradient-boosted trees
(depth 3, heavily subsampled) with pinball-loss quantile heads won because they are
NaN-robust, well regularized at n≈1,100 rows, and produce the predictive distribution the
challenge scores. Linear models collapse under extrapolation (582 µm vs 257 µm baseline);
kNN/forests clamp to the nearest track; a √P physics head was rejected because width vs
power is strongly nonlinear; deep nets are unsupportable at n=4 — and the measured skill
ceiling is label-noise/sensor-bound (55–77% of local variance is noise; ripple dynamics
exceed the 50 fps camera's Nyquist limit), not capacity-bound. Full rationale:
`docs/DATAFLOW_AND_MODEL_TRAINING.md` §2.

## Evaluation protocol

Leave-one-track-out CV over tracks 8/10/14 for every development decision; **Track 21
(200 W) held out and scored once**. Blocked within-track CV (2 mm guard bands) reported
separately. All preprocessing — GBM bin edges, lag, weights, conformal multiplier — is fit
on training folds only. Leakage audit: `docs/FINAL_REPORT.md` §8/§9.

## Requirements

`numpy`, `pandas`, `Pillow`, `matplotlib` (see `requirements.txt`). No scipy/sklearn
needed — the MAT reader and the gradient-boosting model are implemented from scratch in
`src/fmrg_lib.py` / `src/np_models.py` and validated against XGBoost.

## Data & citation

Dataset: NSF Future Manufacturing Data Challenge, Zenodo
[10.5281/zenodo.21285367](https://doi.org/10.5281/zenodo.21285367) (CC-BY-4.0).
Please cite the companion dataset paper (arXiv:2607.07965) and the challenge repository
(github.com/abhishekhanchate/nsf-fmrg-data-challenge). Supported by NSF grant FMRG-2328395.
