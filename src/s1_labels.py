"""STEP 1 — Ground-truth extraction: melt-track geometry from height maps.

These no-powder remelt tracks are nearly flush with the substrate, so height
thresholding fails. The melt zone is instead optically SMOOTH and reflective:
the profilometer returns dense valid data inside it (4-15% NaN) vs sparse data
on rough substrate (60-80% NaN). Score = validity x smoothness; the track is
the largest contiguous above-threshold band per pixel column.

Outputs per track: labels_{tid}.npz with width/yl/yr/edge_rough/valid_frac per
0.2 mm bin (+ per-column curves), and hm_score_{tid}.npy for figures.

Usage:  python s1_labels.py [track_id ...]
"""
import sys
import numpy as np
from fmrg_paths import CACHE, TRACKS

PIX = 0.003982   # mm per height-map pixel
BIN_MM = 0.2


def box_mean(A, ry, rx):
    """Box mean with edge padding via summed-area table."""
    P = np.pad(A, ((ry + 1, ry), (rx + 1, rx)), mode='edge').astype(np.float64)
    P[0, :] = 0
    P[:, 0] = 0
    c = P.cumsum(0).cumsum(1)
    h, w = A.shape
    ky, kx = 2 * ry + 1, 2 * rx + 1
    S = (c[ky:ky + h, kx:kx + w] - c[0:h, kx:kx + w]
         - c[ky:ky + h, 0:w] + c[0:h, 0:w])
    return S / (ky * kx)


def largest_run(mask, bridge=0):
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    runs = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev - 1 > bridge:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return max(runs, key=lambda r: r[1] - r[0])


def melt_score(Z):
    """Validity x smoothness score map."""
    V = np.isfinite(Z).astype(np.float32)
    dz = np.abs(np.diff(np.where(np.isfinite(Z), Z, np.nan), axis=0)) * 1e3  # um/px
    dz = np.vstack([dz, dz[-1:]])
    R = np.where(np.isfinite(dz), dz, 8.0)   # NaN => rough
    Vs = box_mean(V, 7, 25)
    Rs = box_mean(np.minimum(R, 8.0), 7, 25)
    return Vs * np.exp(-Rs / 2.0)


def extract(tid, thr_mult=1.0, bridge=12, save=True, score=None):
    Z = np.load(CACHE / f'hm_Z_{tid}.npy')
    x = np.load(CACHE / f'hm_x_{tid}.npy')
    y = np.load(CACHE / f'hm_y_{tid}.npy')
    if score is None:
        score = melt_score(Z)
    thr = 0.5 * (np.percentile(score, 85) + np.percentile(score, 30)) * thr_mult

    n = Z.shape[1]
    width = np.full(n, np.nan)
    yl = np.full(n, np.nan)
    yr = np.full(n, np.nan)
    for j in range(n):
        r = largest_run(score[:, j] > thr, bridge=bridge)
        if r is None:
            continue
        s, e = r
        w = (e - s + 1) * PIX
        if w < 0.15:            # too narrow to be the melt track
            continue
        width[j] = w
        yl[j] = y[s]
        yr[j] = y[e]

    edges = np.arange(20.0, 100.0 + 1e-9, BIN_MM)
    xc = 0.5 * (edges[:-1] + edges[1:])
    bi = np.digitize(x, edges) - 1
    out = {'x': xc}
    for k in ['width', 'yl', 'yr', 'edge_rough', 'valid_frac']:
        out[k] = np.full(len(xc), np.nan)
    for b in range(len(xc)):
        m = bi == b
        if not m.any():
            continue
        w = width[m]
        v = np.isfinite(w)
        out['valid_frac'][b] = v.mean()
        if v.sum() < 8:
            continue
        out['width'][b] = np.median(w[v])
        out['yl'][b] = np.median(yl[m][v])
        out['yr'][b] = np.median(yr[m][v])
        out['edge_rough'][b] = 0.5 * (np.std(yl[m][v]) + np.std(yr[m][v]))
    if save:
        np.savez(CACHE / f'labels_{tid}.npz', **out,
                 col_width=width, col_yl=yl, col_yr=yr, col_x=x, score_thr=thr)
        np.save(CACHE / f'hm_score_{tid}.npy', score[:, ::10].astype(np.float32))
        fin = np.isfinite(out['width'])
        print(f'  labels T{tid}: {fin.sum()}/400 bins, '
              f'width med={np.nanmedian(out["width"])*1e3:.0f} um')
    return out


if __name__ == '__main__':
    for t in [int(a) for a in sys.argv[1:]] or TRACKS:
        extract(t)
