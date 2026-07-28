"""Final model: blend(GBM, physics-anchor linear on th_a1400_m5) + quantiles + conformal.
All choices (lag, anchor feature, blend weight, conformal multiplier) made on train tracks only.
"""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from np_models import HistGBM, KNN, mae, crps_from_quantiles, coverage

import fmrg_paths as _P
CACHE = str(_P.CACHE)
DATA = str(_P.DATA)
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
TR = [8, 10, 14]
ANCHOR = 'th_a1400_m5'   # selected via LOTO on training tracks
LAG = 14                 # selected via training tracks


def load():
    D = pd.read_csv(f'{CACHE}/datasetW.csv')
    D = D[D['width'].notna()].reset_index(drop=True)
    th_cols = [c for c in D.columns if c.startswith('th_')]
    for t in D['track'].unique():
        m = (D['track'] == t).values
        for c in th_cols:
            D.loc[m, c] = pd.Series(D.loc[m, c].values).shift(LAG).values
    drop = ['track', 'x', 'width', 'edge_rough', 'valid_frac', 'width_iqr']
    cols = [c for c in D.columns if c not in drop]
    return D, cols


def anchor_pred(tr, te):
    x = tr[ANCHOR].values
    y = tr['width'].values
    m = np.isfinite(x) & np.isfinite(y)
    b, a = np.polyfit(x[m], y[m], 1)
    xx = te[ANCHOR].values
    return a + b * np.nan_to_num(xx, nan=np.nanmedian(xx))


def fold(D, cols, held, with_q=False):
    trt = [t for t in (TR if held in TR else TR) if t != held]
    tr = D[D['track'].isin(trt)]
    te = D[D['track'] == held]
    X, y = tr[cols].values.astype(float), tr['width'].values.astype(float)
    Xte, yte = te[cols].values.astype(float), te['width'].values.astype(float)
    vf = tr['valid_frac'].values
    wgt = np.clip(vf, 0.05, None)
    g = HistGBM(n_trees=150, lr=0.08, depth=3, seed=0).fit(X, y, sample_weight=wgt)
    pg = g.predict(Xte)
    pa = anchor_pred(tr, te)
    out = {'y': yte, 'gbm': pg, 'anchor': pa, 'x': te['x'].values}
    if with_q:
        Qs = []
        for q in QS:
            mq = HistGBM(n_trees=100, lr=0.08, depth=3, loss='pinball', q=q, seed=0).fit(X, y)
            Qs.append(mq.predict(Xte))
        out['Q'] = np.sort(np.stack(Qs), axis=0)
    return out


def main(mode):
    D, cols = load()
    if mode in ('8', '10', '14'):
        held = int(mode)
        r = fold(D, cols, held, with_q=True)
        np.savez(f'{CACHE}/blend_f{held}.npz', **r)
        for al in [0.0, 0.25, 0.5, 0.75, 1.0]:
            p = al * r['gbm'] + (1 - al) * r['anchor']
            print(f'held {held} alpha={al}: MAE={mae(r["y"], p):.0f}')
        return
    if mode == 'final':
        # choose alpha on train folds
        maes = {}
        conf_z = []
        for al in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ms = []
            for held in TR:
                r = np.load(f'{CACHE}/blend_f{held}.npz')
                p = al * r['gbm'] + (1 - al) * r['anchor']
                ms.append(mae(r['y'], p))
            maes[al] = np.mean(ms)
        alpha = min(maes, key=maes.get)
        print('train-LOTO MAE by alpha:', {k: round(v) for k, v in maes.items()}, '-> alpha =', alpha)
        # conformal scores from train folds using blended center
        for held in TR:
            r = np.load(f'{CACHE}/blend_f{held}.npz')
            p = alpha * r['gbm'] + (1 - alpha) * r['anchor']
            Q = r['Q'] + (p - r['Q'][2])[None, :]
            half = np.maximum((Q[4] - Q[0]) / 2, 1e-9)
            conf_z.extend((np.abs(r['y'] - p) / half).tolist())
        s90 = float(np.quantile(conf_z, 0.90))
        print(f'conformal multiplier s90={s90:.2f}')
        # TEST on track 21
        r = fold(D, cols, 21, with_q=True)
        p = alpha * r['gbm'] + (1 - alpha) * r['anchor']
        Q = r['Q'] + (p - r['Q'][2])[None, :]
        Qc = p[None, :] + (Q - p[None, :]) * s90
        res = dict(alpha=alpha, s90=s90,
                   mae_gbm=mae(r['y'], r['gbm']), mae_anchor=mae(r['y'], r['anchor']),
                   mae_blend=mae(r['y'], p),
                   crps_cal=crps_from_quantiles(r['y'], Qc, QS),
                   cov90_cal=coverage(r['y'], Qc[0], Qc[4]),
                   cov50_cal=coverage(r['y'], Qc[1], Qc[3]),
                   local_corr=float(np.corrcoef(p - p.mean(), r['y'] - r['y'].mean())[0, 1]))
        print('TEST 21:', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()})
        np.savez(f'{CACHE}/blend_test21.npz', y=r['y'], x=r['x'], pred=p, Q=Qc)
        json.dump(res, open(f'{CACHE}/blend_final.json', 'w'))


if __name__ == '__main__':
    main(sys.argv[1])
