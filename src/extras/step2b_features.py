"""Thermal melt-pool features + SEM substrate-morphology features per 0.2mm x-bin."""
import sys, glob
import numpy as np
from PIL import Image

import fmrg_paths as _P
CACHE = str(_P.CACHE)
DATA = str(_P.DATA)
THRESHOLDS = [1400, 1600, 1800, 2000, 2200]
PX_MM = 0.014  # thermal pixel size


def thermal_features(tid):
    seg = np.load(f'{CACHE}/th_seg_{tid}.npy')  # (400, 400, 400) frames
    x_mm = np.load(f'{CACHE}/th_x_{tid}.npy')
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
        feats[f'a{thr}'] = area * PX_MM ** 2  # mm^2
        # moments (guard empty)
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
            feats[f'len{thr}'] = 4 * sx * PX_MM   # along-scan extent proxy
            feats[f'wid{thr}'] = 4 * sy * PX_MM   # cross-scan extent proxy
            with np.errstate(invalid='ignore', divide='ignore'):
                feats[f'ecc{thr}'] = sx / np.maximum(sy, 1e-9)
                feats[f'skewx{thr}'] = sxy / np.maximum(sx * sy, 1e-9)
    # trailing tail: column-profile of frame max over y, extent above 1200 behind pool center
    prof = seg.max(1)  # (n, 400) max over rows -> profile along scan axis
    tail = np.full(n, np.nan)
    grad = np.full(n, np.nan)
    for i in range(n):
        p = prof[i]
        pk = int(np.argmax(p))
        above = p > 1200
        # extent of contiguous above-threshold region through the peak
        l = pk
        while l > 0 and above[l - 1]:
            l -= 1
        r = pk
        while r < 399 and above[r + 1]:
            r += 1
        tail[i] = (r - l + 1) * PX_MM
        seg10 = p[max(0, pk - 40):pk]
        if len(seg10) > 5:
            grad[i] = np.polyfit(np.arange(len(seg10)), seg10, 1)[0]
    feats['tail_len'] = tail
    feats['tail_grad'] = grad
    feats['x'] = x_mm
    np.savez(f'{CACHE}/thfeat_{tid}.npz', **feats)
    print(f'thermal {tid}: {len(feats)-1} base features, frames {n}')


def sem_features(tid):
    files = sorted(glob.glob(f'{DATA}/sem/SEM_{tid}/PlainImages/*.tif'),
                   key=lambda p: int(p.split('_')[-1].split('.')[0]))
    tiles = [np.asarray(Image.open(f).convert('L'), np.float32) for f in files]
    ntile = len(tiles)
    TILE_MM = 6.41
    H, W = tiles[0].shape
    px_mm = TILE_MM / W  # ~6.26 um/px

    # find track band per tile: row-smoothness (track = low local gradient variance)
    def track_band(im):
        g = np.abs(np.diff(im, axis=1))
        rowvar = g.mean(1)
        k = 51
        rv = np.convolve(rowvar, np.ones(k) / k, 'same')
        c = int(np.argmin(rv))
        return c
    # features per 0.2mm bin
    edges = np.arange(20.0, 100.0 + 1e-9, 0.2)
    xc = 0.5 * (edges[:-1] + edges[1:])
    names = ['s_mean', 's_std', 's_grad', 's_locvar', 's_fftlow', 's_ffthigh', 's_entropy']
    out = {k: np.full(len(xc), np.nan) for k in names}
    band_half = int(0.75 / px_mm)  # exclude central +/-0.45mm around track
    for b, x0 in enumerate(xc):
        # tile index: tile k (0-based) covers [100-(k+1)*6.41, 100-k*6.41]
        k = int((100.0 - x0) // TILE_MM)
        if k < 0 or k >= ntile:
            continue
        im = tiles[k]
        # column within tile: x increases opposite tile order
        col_f = (100.0 - k * TILE_MM - x0) / TILE_MM * W
        c0, c1 = int(col_f - 0.1 / px_mm), int(col_f + 0.1 / px_mm)
        c0, c1 = max(0, c0), min(W, c1)
        if c1 - c0 < 8:
            continue
        tc = track_band(im)
        rows = np.r_[np.arange(0, max(0, tc - band_half)),
                     np.arange(min(H, tc + band_half), H)]
        if rows.size < 50:
            continue
        patch = im[np.ix_(rows, np.arange(c0, c1))]
        out['s_mean'][b] = patch.mean()
        out['s_std'][b] = patch.std()
        g = np.abs(np.diff(patch, axis=0))
        out['s_grad'][b] = g.mean()
        # local variance (texture energy)
        mu = patch.mean(0)
        out['s_locvar'][b] = ((patch - mu) ** 2).mean()
        # fft along y (ripples across track direction)
        p = patch - patch.mean()
        F = np.abs(np.fft.rfft(p, axis=0)).mean(1)
        nn = len(F)
        out['s_fftlow'][b] = F[1:nn // 8].mean()
        out['s_ffthigh'][b] = F[nn // 2:].mean()
        h, _ = np.histogram(patch, bins=32, range=(0, 255), density=True)
        h = h[h > 0]
        out['s_entropy'][b] = -(h * np.log(h)).sum()
    out['x'] = xc
    np.savez(f'{CACHE}/semfeatW_{tid}.npz', **out)
    ok = np.isfinite(out['s_mean']).sum()
    print(f'sem {tid}: {ntile} tiles, {ok}/400 bins with features')


if __name__ == '__main__':
    mode = sys.argv[1]
    tid = int(sys.argv[2])
    if mode == 'th':
        thermal_features(tid)
    else:
        sem_features(tid)
