"""STEP 4 — Assemble the modeling table.

Joins labels + thermal features (with +/-1 mm rolling stats) + SEM features +
power on the common 0.2 mm grid. 400 bins x 4 tracks = 1,600 rows.

Writes cache/dataset.csv.

Usage:  python s4_dataset.py
"""
import numpy as np
import pandas as pd
from fmrg_paths import CACHE, TRACKS, POWER


def rolling(a, w=5, fn=np.nanmean):
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        s = max(0, i - w)
        e = min(len(a), i + w + 1)
        seg = a[s:e]
        if np.isfinite(seg).sum() >= 3:
            out[i] = fn(seg)
    return out


def build():
    rows = []
    for tid in TRACKS:
        lab = np.load(CACHE / f'labels_{tid}.npz')
        th = np.load(CACHE / f'thfeat_{tid}.npz')
        se = np.load(CACHE / f'semfeat_{tid}.npz')
        df = pd.DataFrame({
            'track': tid, 'power': POWER[tid], 'x': lab['x'],
            'width': lab['width'] * 1e3,          # um
            'yl': lab['yl'] * 1e3,
            'yr': lab['yr'] * 1e3,
            'edge_rough': lab['edge_rough'] * 1e3,
            'valid_frac': lab['valid_frac'],
        })
        df['center'] = (df['yl'] + df['yr']) / 2
        for k in th.files:
            if k == 'x':
                continue
            v = th[k].astype(np.float64)
            df[f'th_{k}'] = v
            df[f'th_{k}_m5'] = rolling(v, 5, np.nanmean)
            df[f'th_{k}_s5'] = rolling(v, 5, np.nanstd)
        for k in se.files:
            if k == 'x':
                continue
            df[f'sem_{k}'] = se[k].astype(np.float64)
        rows.append(df)
    D = pd.concat(rows, ignore_index=True)
    D.to_csv(CACHE / 'dataset.csv', index=False)
    print(f'  dataset: {D.shape}, labeled rows {D["width"].notna().sum()}')
    g = D.groupby('power')['width'].median()
    print('  sanity (median width by power, should increase):')
    for p, w in g.items():
        print(f'    {p} W -> {w:.0f} um')


if __name__ == '__main__':
    build()
