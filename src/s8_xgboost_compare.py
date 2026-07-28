"""STEP 8 (optional) — XGBoost head-to-head on identical folds.

Requires `pip install xgboost scipy`. Skipped gracefully if not installed.

Usage:  python s8_xgboost_compare.py
"""
import json
import numpy as np
import pandas as pd
from fmrg_paths import CACHE, TRAIN_TRACKS, TEST_TRACK
from np_models import mae, crps_from_quantiles, coverage
from s7_final_model import load, QS

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit('xgboost not installed - `pip install xgboost` to run this '
                     'comparison (optional; the main pipeline does not need it).')

PARAMS = dict(max_depth=3, eta=0.08, subsample=0.9, colsample_bytree=0.9,
              tree_method='hist', max_bin=32, reg_lambda=1.0)


def run():
    D, cols = load()
    X = D[cols].values.astype(np.float32)
    y = D['w_ens'].values.astype(np.float32)
    wgt = np.clip(D['valid_frac'].values, 0.05, None) / \
        (1 + np.nan_to_num(D['w_spread'].values, nan=30) / 30.0)
    results = {}
    for held in TRAIN_TRACKS + [TEST_TRACK]:
        tr = (D['track'].isin([t for t in TRAIN_TRACKS if t != held])
              if held in TRAIN_TRACKS else D['track'].isin(TRAIN_TRACKS)).values
        te = (D['track'] == held).values
        dtr = xgb.DMatrix(X[tr], label=y[tr], weight=wgt[tr])
        dte = xgb.DMatrix(X[te])
        m = xgb.train({**PARAMS, 'objective': 'reg:squarederror'}, dtr, num_boost_round=150)
        p = m.predict(dte)
        Q = []
        for q in QS:
            mq = xgb.train({**PARAMS, 'objective': 'reg:quantileerror',
                            'quantile_alpha': q}, dtr, num_boost_round=100)
            Q.append(mq.predict(dte))
        Q = np.sort(np.stack(Q), axis=0)
        Qc = Q + (p - Q[2])[None, :]
        results[str(held)] = dict(mae=float(mae(y[te], p)),
                                  crps=float(crps_from_quantiles(y[te], Qc, QS)),
                                  cov90=float(coverage(y[te], Qc[0], Qc[4])))
        print(f'  xgb held={held}:', results[str(held)])
    with open(CACHE / 'xgb_results.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    run()
