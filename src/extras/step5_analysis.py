"""Feature importance, within-track correlations, lag analysis."""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from np_models import HistGBM

import fmrg_paths as _P
CACHE = str(_P.CACHE)
DATA = str(_P.DATA)


def main():
    D = pd.read_csv(f'{CACHE}/dataset.csv')
    D = D[D['width'].notna()].reset_index(drop=True)
    drop = ['track', 'x', 'width', 'edge_rough', 'valid_frac']
    cols = [c for c in D.columns if c not in drop]
    X = D[cols].values.astype(np.float64)
    y = D['width'].values.astype(np.float64)

    # feature importance: GBM trained on tracks 8,10,14 (challenge train split)
    tr = D['track'].isin([8, 10, 14]).values
    m = HistGBM(n_trees=200, lr=0.08, depth=3, seed=1).fit(X[tr], y[tr])
    imp = m.feature_importance(X.shape[1])
    order = np.argsort(imp)[::-1]
    top = [(cols[i], float(imp[i])) for i in order[:20]]
    print('TOP-20 GBM feature importance (train 8/10/14):')
    for n_, v in top:
        print(f'  {n_:28s} {v:.3f}')

    # within-track (local) spearman-ish correlations for key features
    def demean(s, t):
        return s - s[t].mean()
    feats_check = [c for c in cols if c != 'power']
    loc_corr = {}
    for c in feats_check:
        v = D[c].values
        acc = []
        for t in [8, 10, 14, 21]:
            mt = (D['track'] == t).values & np.isfinite(v) & np.isfinite(y)
            if mt.sum() < 50:
                continue
            vv = v[mt] - v[mt].mean()
            yy = y[mt] - y[mt].mean()
            if vv.std() < 1e-12:
                continue
            acc.append(np.corrcoef(vv, yy)[0, 1])
        if acc:
            loc_corr[c] = float(np.mean(acc))
    ranked = sorted(loc_corr.items(), key=lambda kv: -abs(kv[1]))
    print('\nTOP-15 |within-track corr| with width:')
    for c, r in ranked[:15]:
        print(f'  {c:28s} {r:+.3f}')

    # lag analysis on best local feature per track
    print('\nLag analysis (bins of 0.2mm), feature th_a1800_m5:')
    v = D['th_a1800_m5'].values
    for t in [8, 10, 14, 21]:
        mt = (D['track'] == t).values
        vv = v[mt]
        yy = y[mt]
        fin = np.isfinite(vv) & np.isfinite(yy)
        vv = np.where(fin, vv, np.nan)
        best = (0, 0.0)
        for lag in range(-15, 16):
            a = pd.Series(vv).shift(lag).values
            m2 = np.isfinite(a) & np.isfinite(yy)
            if m2.sum() < 100:
                continue
            aa = a[m2] - a[m2].mean()
            bb = yy[m2] - yy[m2].mean()
            r = np.corrcoef(aa, bb)[0, 1]
            if abs(r) > abs(best[1]):
                best = (lag, r)
        print(f'  track {t}: best lag {best[0]:+d} bins ({best[0]*0.2:+.1f}mm) r={best[1]:+.2f}')

    json.dump({'importance': top, 'local_corr': ranked[:25]},
              open(f'{CACHE}/analysis.json', 'w'))


if __name__ == '__main__':
    main()
