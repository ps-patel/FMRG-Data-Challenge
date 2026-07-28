"""STEP 5 — Label-extraction ensemble: quantify ground-truth sensitivity.

Re-runs the s1 extractor over a 6-point hyperparameter grid (threshold x
{0.85, 1.0, 1.15}, NaN-bridge {8, 16} px), then merges the ensemble-median
label + per-bin spread into the dataset.

Writes cache/dataset_ensemble.csv  (adds w_ens, w_spread, w_v0..w_v5).

Usage:  python s5_label_ensemble.py
"""
import numpy as np
import pandas as pd
from fmrg_paths import CACHE, TRACKS
from s1_labels import extract, melt_score

VARIANTS = [(tm, br) for tm in (0.85, 1.0, 1.15) for br in (8, 16)]


def main():
    D = pd.read_csv(CACHE / 'dataset.csv')
    for tid in TRACKS:
        score = melt_score(np.load(CACHE / f'hm_Z_{tid}.npy'))  # compute once
        W = []
        for tm, br in VARIANTS:
            out = extract(tid, thr_mult=tm, bridge=br, save=False, score=score)
            W.append(out['width'] * 1e3)
        W = np.stack(W)
        m = (D['track'] == tid).values
        D.loc[m, 'w_ens'] = np.nanmedian(W, 0)
        D.loc[m, 'w_spread'] = np.nanstd(W, 0)
        for v in range(len(VARIANTS)):
            D.loc[m, f'w_v{v}'] = W[v]
        print(f'  ensemble T{tid}: median width '
              f'{np.nanmedian(W):.0f} um, spread med '
              f'{np.nanmedian(np.nanstd(W,0)):.0f} um')
    D.to_csv(CACHE / 'dataset_ensemble.csv', index=False)
    d = D[D['w_ens'].notna()]
    print(f'  per-bin label spread: median={d["w_spread"].median():.0f} um, '
          f'p90={d["w_spread"].quantile(.9):.0f} um')


if __name__ == '__main__':
    main()
