# Extras — research scripts from the analysis

These are the investigation scripts (several document deliberately NEGATIVE results —
see FINAL_REPORT.md section 8). They are kept for transparency and are NOT part of the
main pipeline. Notes:

- dataset filenames evolved during the analysis: where a script reads `datasetW.csv` or
  `datasetE.csv`, use `dataset.csv` / `dataset_ensemble.csv` from the main pipeline.
- run them from this folder after the main pipeline has populated the cache.

| Script | Question | Outcome |
|---|---|---|
| step5_analysis.py | feature importance, lag scan | found +2.8 mm misalignment |
| step7_final.py | sqrt(P) physics head | REJECTED - width vs P nonlinear |
| step8_blend.py | anchor/GBM blending | alpha=1.0 (pure GBM won) |
| step9_selfalign.py | per-track self-alignment | alignment not the bottleneck |
| step10_spectral.py | melt-pool oscillation spectra | no transfer (Nyquist limit) |
| step13_stitch_xcorr.py | seam cross-correlation stitching | unreliable, kept 5% spec |
| step2b_features.py | leak-fixed tile-wise SEM features | superseded by s3 mosaic |
