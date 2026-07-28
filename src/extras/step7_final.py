"""Improvement pipeline: (1) lag alignment, (2) sqrt(P) physics head,
(3) label-quality weights, (4) conformal calibration, (5) ensemble."""
import sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from np_models import HistGBM, KNN, mae, pinball, crps_from_quantiles, coverage

import fmrg_paths as _P
CACHE = str(_P.CACHE)
DATA = str(_P.DATA)
QS = [0.05, 0.25, 0.5, 0.75, 0.95]
TR_TRACKS = [8, 10, 14]


def load():
    D = pd.read_csv(f'{CACHE}/datasetW.csv')
    return D


def pick_lag(D, tracks):
    """Best common lag (bins) maximizing mean |within-track corr| of thermal
    features with width, using TRAINING tracks only."""
    th_cols = [c for c in D.columns if c.startswith('th_') and not c.endswith('_s5')]
    best = (0, -1)
    for lag in range(-16, 17):
        accs = []
        for t in tracks:
            d = D[D['track'] == t]
            y = d['width'].values
            cors = []
            for c in th_cols:
                v = pd.Series(d[c].values).shift(lag).values
                m = np.isfinite(v) & np.isfinite(y)
                if m.sum() < 100:
                    continue
                vv = v[m] - v[m].mean()
                yy = y[m] - y[m].mean()
                if vv.std() < 1e-12 or yy.std() < 1e-12:
                    continue
                cors.append(abs(np.corrcoef(vv, yy)[0, 1]))
            if cors:
                accs.append(np.mean(np.sort(cors)[-10:]))  # mean of top-10
        score = np.mean(accs)
        if score > best[1]:
            best = (lag, score)
    return best


def apply_lag(D, lag):
    D2 = D.copy()
    th_cols = [c for c in D.columns if c.startswith('th_')]
    for t in D['track'].unique():
        m = (D['track'] == t).values
        for c in th_cols:
            D2.loc[m, c] = pd.Series(D.loc[m, c].values).shift(lag).values
    return D2


def physics_w0(D, tracks):
    """Fit width = a + b*sqrt(P) on track medians of TRAINING tracks."""
    meds = D[D['track'].isin(tracks)].groupby('power')['width'].median()
    A = np.c_[np.sqrt(meds.index.values), np.ones(len(meds))]
    coef, *_ = np.linalg.lstsq(A, meds.values, rcond=None)
    return lambda P: coef[0] * np.sqrt(P) + coef[1]


def features(D):
    drop = ['track', 'x', 'width', 'edge_rough', 'valid_frac', 'power']
    cols = [c for c in D.columns if c not in drop] + ['power']
    return cols


def run(out_json, folds=None):
    D0 = load()
    D0 = D0[D0['width'].notna()].reset_index(drop=True)

    # ---- step 1: alignment (training tracks only) ----
    import os
    if os.path.exists(f'{CACHE}/lag.json'):
        j = json.load(open(f'{CACHE}/lag.json'))
        lag, score = j['lag'], j['score']
    else:
        lag, score = pick_lag(D0, TR_TRACKS)
    D = apply_lag(D0, lag)
    print(f'[1] chosen lag {lag:+d} bins ({lag*0.2:+.1f} mm), top10 |corr| {score:.3f}')

    cols = features(D)
    X = D[cols].values.astype(float)
    y = D['width'].values.astype(float)
    tr_ids = D['track'].values
    P = D['power'].values.astype(float)

    # ---- step 3: quality weights ----
    iqr = D['width_iqr'].values if 'width_iqr' in D else np.zeros(len(D))
    vf = D['valid_frac'].values
    wgt = vf / (1.0 + np.nan_to_num(iqr, nan=50) / 50.0)
    wgt = np.clip(wgt, 0.05, None)

    results = {'lag': lag}
    # conformal scores accumulated over LOTO folds
    conf_z = []
    fold_rows = {}
    folds = folds or (TR_TRACKS + [21])
    for held in folds:
        if held == 21:
            tr = np.isin(tr_ids, TR_TRACKS)
        else:
            tr = np.isin(tr_ids, [t for t in TR_TRACKS if t != held])
        te = tr_ids == held
        w0 = physics_w0(D[tr], sorted(set(tr_ids[tr])))
        r_tr = y[tr] - w0(P[tr])
        r_te_true = y[te] - w0(P[te])

        # ---- step 2+3: residual models with weights ----
        g = HistGBM(n_trees=150, lr=0.08, depth=3, seed=0).fit(X[tr], r_tr, sample_weight=wgt[tr])
        k = KNN(15).fit(X[tr], r_tr)
        pg = g.predict(X[te])
        pk = k.predict(X[te])
        pe = 0.5 * (pg + pk)                       # ---- step 5: ensemble ----
        pred = w0(P[te]) + pe
        res = dict(mae_gbm=mae(y[te], w0(P[te]) + pg),
                   mae_knn=mae(y[te], w0(P[te]) + pk),
                   mae_ens=mae(y[te], pred),
                   mae_physics_only=mae(y[te], w0(P[te])))
        # quantile residual models
        Q = []
        for q in QS:
            mq = HistGBM(n_trees=100, lr=0.08, depth=3, loss='pinball', q=q, seed=0)
            mq.fit(X[tr], r_tr)
            Q.append(mq.predict(X[te]))
        Q = np.sort(np.stack(Q), axis=0)
        # recentre quantiles on ensemble median
        Qc = Q + (pe - Q[2])[None, :]
        yy = y[te]
        qabs = w0(P[te])[None, :] + Qc
        res['crps_raw'] = crps_from_quantiles(yy, qabs, QS)
        res['cov90_raw'] = coverage(yy, qabs[0], qabs[4])
        if held != 21:
            half = np.maximum((qabs[4] - qabs[0]) / 2, 1e-9)
            conf_z.extend((np.abs(yy - qabs[2]) / half).tolist())
        fold_rows[held] = dict(q=qabs.tolist(), y=yy.tolist(),
                               x=D.loc[te, 'x'].values.tolist(), res=res)
        print(f'held {held}: physics-only mae={res["mae_physics_only"]:.0f} '
              f'gbm={res["mae_gbm"]:.0f} ens={res["mae_ens"]:.0f} '
              f'crps={res["crps_raw"]:.0f} cov90={res["cov90_raw"]:.2f}')

    # per-fold mode: dump partial state and exit
    if len(folds) < 4:
        json.dump({'conf_z': conf_z,
                   'fold_rows': fold_rows}, open(out_json, 'w'))
        return

    # ---- step 4: conformal multiplier from LOTO folds (90% target) ----
    s90 = float(np.quantile(conf_z, 0.90))
    print(f'[4] conformal width multiplier s90={s90:.2f}')
    fold = fold_rows[21]
    qa = np.array(fold['q'])
    yy = np.array(fold['y'])
    center = qa[2]
    qa_cal = center[None, :] + (qa - center[None, :]) * s90
    fold['res']['crps_cal'] = crps_from_quantiles(yy, qa_cal, QS)
    fold['res']['cov90_cal'] = coverage(yy, qa_cal[0], qa_cal[4])
    fold['q_cal'] = qa_cal.tolist()
    print(f'TEST 21 after calibration: crps={fold["res"]["crps_cal"]:.0f} '
          f'cov90={fold["res"]["cov90_cal"]:.2f} (raw {fold["res"]["cov90_raw"]:.2f})')

    # local skill after alignment (ensemble, blocked style quick check):
    loc = {}
    for t in [8, 10, 14, 21]:
        r = fold_rows.get(t)
        if r is None:
            continue
        p = np.array(r['q'])[2]
        yy2 = np.array(r['y'])
        loc[t] = float(np.corrcoef(p - p.mean(), yy2 - yy2.mean())[0, 1])
    print('[1] local corr (median pred, demeaned):', {k: round(v, 2) for k, v in loc.items()})
    results['folds'] = {k: v['res'] for k, v in fold_rows.items()}
    results['local_corr'] = loc
    results['s90'] = s90
    json.dump({'results': results, 'test21': fold_rows[21]}, open(out_json, 'w'))


def combine(parts, out_json):
    conf_z = []
    fold_rows = {}
    for p in parts:
        j = json.load(open(p))
        conf_z.extend(j['conf_z'])
        fold_rows.update({int(k): v for k, v in j['fold_rows'].items()})
    s90 = float(np.quantile(conf_z, 0.90))
    print(f'[4] conformal width multiplier s90={s90:.2f}')
    fold = fold_rows[21]
    qa = np.array(fold['q'])
    yy = np.array(fold['y'])
    center = qa[2]
    qa_cal = center[None, :] + (qa - center[None, :]) * s90
    fold['res']['crps_cal'] = crps_from_quantiles(yy, qa_cal, QS)
    fold['res']['cov90_cal'] = coverage(yy, qa_cal[0], qa_cal[4])
    fold['q_cal'] = qa_cal.tolist()
    for held, r in sorted(fold_rows.items()):
        print(f"held {held}: physics-only={r['res']['mae_physics_only']:.0f} "
              f"gbm={r['res']['mae_gbm']:.0f} knn={r['res']['mae_knn']:.0f} "
              f"ens={r['res']['mae_ens']:.0f} crps={r['res']['crps_raw']:.0f} "
              f"cov90={r['res']['cov90_raw']:.2f}")
    print(f'TEST 21 calibrated: crps={fold["res"]["crps_cal"]:.0f} '
          f'cov90={fold["res"]["cov90_cal"]:.2f}')
    loc = {}
    for t, r in fold_rows.items():
        p = np.array(r['q'])[2]
        yy2 = np.array(r['y'])
        loc[t] = float(np.corrcoef(p - p.mean(), yy2 - yy2.mean())[0, 1])
    print('local corr (median pred, demeaned):', {k: round(v, 2) for k, v in sorted(loc.items())})
    json.dump({'s90': s90, 'local_corr': loc,
               'folds': {k: v['res'] for k, v in fold_rows.items()},
               'test21': fold_rows[21]}, open(out_json, 'w'))


if __name__ == '__main__':
    if sys.argv[1] == 'combine':
        combine(sys.argv[3].split(','), sys.argv[2])
    else:
        run(sys.argv[1], [int(f) for f in sys.argv[2].split(',')] if len(sys.argv) > 2 else None)
