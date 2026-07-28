"""STEP 7 — FINAL MODEL: probabilistic width + boundaries (+ z-ripple).

Lag-aligned (+14 bins, chosen on training tracks), quality-weighted quantile
GBM on ensemble labels, CRPS-tuned conformal calibration. Track 21 is the
holdout, scored once.

Usage:
  python s7_final_model.py fold 8      # one LOTO fold (also 10, 14, 21)
  python s7_final_model.py combine     # calibrate + final test-21 scorecard
  python s7_final_model.py all         # run everything sequentially
"""
import json
import sys
import numpy as np
import pandas as pd
from fmrg_paths import CACHE, TRAIN_TRACKS, TEST_TRACK
from np_models import HistGBM, mae, crps_from_quantiles, coverage

QS = [0.05, 0.25, 0.5, 0.75, 0.95]
LAG = 14   # bins (+2.8 mm), selected on training tracks only


def load():
    D = pd.read_csv(CACHE / 'dataset_with_z.csv'
                    if (CACHE / 'dataset_with_z.csv').exists()
                    else CACHE / 'dataset_ensemble.csv')
    D = D[D['w_ens'].notna() & D['center'].notna()].reset_index(drop=True)
    for t in D['track'].unique():
        m = (D['track'] == t).values
        for c in [c for c in D.columns if c.startswith('th_')]:
            D.loc[m, c] = pd.Series(D.loc[m, c].values).shift(LAG).values
    label_cols = (['track', 'x', 'width', 'yl', 'yr', 'center', 'edge_rough',
                   'valid_frac', 'w_ens', 'w_spread', 'z_peak', 'z_mean',
                   'z_valley', 'z_range'] + [f'w_v{v}' for v in range(6)])
    cols = [c for c in D.columns if c not in label_cols]
    return D, cols


def fold(held):
    D, cols = load()
    tr = (D['track'].isin([t for t in TRAIN_TRACKS if t != held])
          if held in TRAIN_TRACKS else D['track'].isin(TRAIN_TRACKS)).values
    te = (D['track'] == held).values
    X = D[cols].values.astype(float)
    w = D['w_ens'].values.astype(float)
    c = D['center'].values.astype(float)
    wgt = np.clip(D['valid_frac'].values, 0.05, None) / \
        (1 + np.nan_to_num(D['w_spread'].values, nan=30) / 30.0)
    Q = []
    for q in QS:
        m = HistGBM(n_trees=100, lr=0.08, depth=3, loss='pinball', q=q, seed=0)
        m.fit(X[tr], w[tr])
        Q.append(m.predict(X[te]))
    Q = np.sort(np.stack(Q), axis=0)
    gw = HistGBM(n_trees=150, lr=0.08, depth=3, seed=0).fit(X[tr], w[tr], sample_weight=wgt[tr])
    gc = HistGBM(n_trees=150, lr=0.08, depth=3, seed=0).fit(X[tr], c[tr], sample_weight=wgt[tr])
    pw = gw.predict(X[te])
    pc = gc.predict(X[te])
    Qc = Q + (pw - Q[2])[None, :]
    extra = {}
    if 'z_range' in D.columns and D['z_range'].notna().any():
        zr = D['z_range'].values.astype(float)
        ok = np.isfinite(zr)
        gz = HistGBM(n_trees=150, lr=0.08, depth=3, seed=0)
        gz.fit(X[tr & ok], zr[tr & ok], sample_weight=wgt[tr & ok])
        extra['zr_pred'] = np.where(te[te], gz.predict(X[te]), np.nan)
        extra['zr_true'] = zr[te]
    np.savez(CACHE / f'final_fold_{held}.npz', x=D.loc[te, 'x'].values,
             w_true=w[te], c_true=c[te],
             yl_true=D.loc[te, 'yl'].values, yr_true=D.loc[te, 'yr'].values,
             Q=Qc, pw=pw, pc=pc, **extra)
    print(f'  fold held={held}: width MAE={mae(w[te], pw):.0f} um, '
          f'center MAE={mae(c[te], pc):.0f} um')


def combine():
    grid = np.arange(0.6, 3.01, 0.05)
    best = None
    for s in grid:
        crps_all, cov_all = [], []
        for held in TRAIN_TRACKS:
            d = np.load(CACHE / f'final_fold_{held}.npz')
            Qs = d['pw'][None, :] + (d['Q'] - d['pw'][None, :]) * s
            crps_all.append(crps_from_quantiles(d['w_true'], Qs, QS))
            cov_all.append(coverage(d['w_true'], Qs[0], Qs[4]))
        if np.mean(cov_all) >= 0.88 and (best is None or np.mean(crps_all) < best[1]):
            best = (float(s), float(np.mean(crps_all)))
    s = best[0]
    d = np.load(CACHE / f'final_fold_{TEST_TRACK}.npz')
    Q, y, p, pc = d['Q'], d['w_true'], d['pw'], d['pc']
    Qs = p[None, :] + (Q - p[None, :]) * s
    res = dict(
        conformal_s=s,
        width_mae=float(mae(y, p)),
        width_crps=float(crps_from_quantiles(y, Qs, QS)),
        width_cov90=float(coverage(y, Qs[0], Qs[4])),
        width_cov50=float(coverage(y, Qs[1], Qs[3])),
        center_mae=float(mae(d['c_true'], pc)),
        yleft_mae=float(mae(d['yl_true'], pc - p / 2)),
        yright_mae=float(mae(d['yr_true'], pc + p / 2)),
    )
    if 'zr_pred' in d.files:
        ok = np.isfinite(d['zr_true']) & np.isfinite(d['zr_pred'])
        res['zrange_mae'] = float(mae(d['zr_true'][ok], d['zr_pred'][ok]))
    np.savez(CACHE / 'final_test21.npz', x=d['x'], y=y, p=p, pc=pc, Q=Qs,
             yl_true=d['yl_true'], yr_true=d['yr_true'])
    with open(CACHE / 'final_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('  FINAL Track-21 scorecard:')
    for k, v in res.items():
        print(f'    {k:14s} {v:.2f}')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode == 'fold':
        fold(int(sys.argv[2]))
    elif mode == 'combine':
        combine()
    else:
        for h in TRAIN_TRACKS + [TEST_TRACK]:
            fold(h)
        combine()
