"""STEP 6 — Z-axis geometry: height descriptors inside the track band.

Per 0.2 mm bin, relative to the detrended substrate plane (z = 0):
z_peak (p98), z_mean, z_valley (p02), z_range (p95-p05, ripple amplitude).

Merges into cache/dataset_with_z.csv.

Usage:  python s6_zprofile.py
"""
import numpy as np
import pandas as pd
from fmrg_paths import CACHE, TRACKS
from fmrg_lib import robust_plane_detrend


def zlabels(tid, D):
    Z = np.load(CACHE / f'hm_Z_{tid}.npy')
    x = np.load(CACHE / f'hm_x_{tid}.npy')
    y = np.load(CACHE / f'hm_y_{tid}.npy')
    d = D[D['track'] == tid].reset_index(drop=True)
    yl = np.nanmedian(d['yl'].values) / 1e3
    yr = np.nanmedian(d['yr'].values) / 1e3
    excl = np.zeros_like(Z, bool)
    excl[(y >= yl - 0.15) & (y <= yr + 0.15), :] = True
    Zd, _ = robust_plane_detrend(Z, x, y, exclude_mask=excl)
    Zum = Zd * 1e3
    edges = np.arange(20.0, 100.0 + 1e-9, 0.2)
    bi = np.digitize(x, edges) - 1
    out = {k: np.full(400, np.nan) for k in ['z_peak', 'z_mean', 'z_valley', 'z_range']}
    for b in range(400):
        row = d.iloc[b]
        if not np.isfinite(row['yl']):
            continue
        ym = (y >= row['yl'] / 1e3) & (y <= row['yr'] / 1e3)
        patch = Zum[np.ix_(ym, bi == b)]
        v = patch[np.isfinite(patch)]
        if v.size < 200:
            continue
        out['z_peak'][b] = np.percentile(v, 98)
        out['z_mean'][b] = np.mean(v)
        out['z_valley'][b] = np.percentile(v, 2)
        out['z_range'][b] = np.percentile(v, 95) - np.percentile(v, 5)
    print(f'  z T{tid}: z_mean med={np.nanmedian(out["z_mean"]):+.1f} um, '
          f'z_range med={np.nanmedian(out["z_range"]):.1f} um')
    return out


def main():
    D = pd.read_csv(CACHE / 'dataset_ensemble.csv')
    for tid in TRACKS:
        z = zlabels(tid, D)
        m = (D['track'] == tid).values
        for k, v in z.items():
            if k not in D.columns:
                D[k] = np.nan
            D.loc[m, k] = v
    D.to_csv(CACHE / 'dataset_with_z.csv', index=False)
    print('  written dataset_with_z.csv')


if __name__ == '__main__':
    main()
