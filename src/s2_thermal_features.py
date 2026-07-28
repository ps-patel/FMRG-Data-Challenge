"""STEP 2 — Thermal melt-pool features: 23 physical descriptors per frame.

Reads th_seg_{tid}.npy from step 0; writes thfeat_{tid}.npz (one value per
0.2 mm bin, 400 bins).

Usage:  python s2_thermal_features.py [track_id ...]
"""
import sys
import numpy as np
from fmrg_paths import CACHE, TRACKS

THRESHOLDS = [1400, 1600, 1800, 2000, 2200]
PX_MM = 0.014   # thermal pixel size, mm


def thermal_features(tid):
    seg = np.load(CACHE / f'th_seg_{tid}.npy')
    x_mm = np.load(CACHE / f'th_x_{tid}.npy')
    n = seg.shape[0]
    feats = {}
    flat = seg.reshape(n, -1)
    feats['t_max'] = flat.max(1)
    feats['t_p999'] = np.percentile(flat, 99.9, axis=1)
    feats['t_p99'] = np.percentile(flat, 99, axis=1)
    feats['t_sum_above1000'] = np.where(flat > 1000, flat - 1000, 0).sum(1) / 1e4

    yy, xx = np.mgrid[0:seg.shape[1], 0:seg.shape[2]]
    yy = yy.ravel()
    xx = xx.ravel()
    for thr in THRESHOLDS:
        M = flat > thr
        area = M.sum(1)
        feats[f'a{thr}'] = area * PX_MM ** 2
        cx = np.full(n, np.nan)
        cy = np.full(n, np.nan)
        sx = np.full(n, np.nan)
        sy = np.full(n, np.nan)
        sxy = np.full(n, np.nan)
        for i in range(n):
            m = M[i]
            if area[i] < 5:
                continue
            xs_, ys_ = xx[m], yy[m]
            cx[i], cy[i] = xs_.mean(), ys_.mean()
            sx[i] = xs_.std()
            sy[i] = ys_.std()
            sxy[i] = ((xs_ - cx[i]) * (ys_ - cy[i])).mean()
        if thr in (1400, 1800):
            feats[f'cx{thr}'] = cx * PX_MM
            feats[f'cy{thr}'] = cy * PX_MM
            feats[f'len{thr}'] = 4 * sx * PX_MM
            feats[f'wid{thr}'] = 4 * sy * PX_MM
            with np.errstate(invalid='ignore', divide='ignore'):
                feats[f'ecc{thr}'] = sx / np.maximum(sy, 1e-9)
                feats[f'skewx{thr}'] = sxy / np.maximum(sx * sy, 1e-9)

    prof = seg.max(1)   # scan-axis profile of frame max
    tail = np.full(n, np.nan)
    grad = np.full(n, np.nan)
    for i in range(n):
        p = prof[i]
        pk = int(np.argmax(p))
        above = p > 1200
        lo = pk
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = pk
        while hi < len(p) - 1 and above[hi + 1]:
            hi += 1
        tail[i] = (hi - lo + 1) * PX_MM
        seg10 = p[max(0, pk - 40):pk]
        if len(seg10) > 5:
            grad[i] = np.polyfit(np.arange(len(seg10)), seg10, 1)[0]
    feats['tail_len'] = tail
    feats['tail_grad'] = grad
    feats['x'] = x_mm
    np.savez(CACHE / f'thfeat_{tid}.npz', **feats)
    print(f'  thermal features T{tid}: {len(feats)-1} descriptors x {n} bins')


if __name__ == '__main__':
    for t in [int(a) for a in sys.argv[1:]] or TRACKS:
        thermal_features(t)
