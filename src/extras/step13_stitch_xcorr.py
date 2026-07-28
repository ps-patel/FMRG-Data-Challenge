"""Refined SEM stitching: estimate the true overlap of each tile seam by
cross-correlating the tiles' overlapping strips (instead of assuming 5%)."""
import sys, glob
import numpy as np
from PIL import Image

import fmrg_paths as _P
CACHE = str(_P.CACHE)
DATA = str(_P.DATA)
TILE_MM = 6.41


def seam_offset(a, b, max_ovl=0.12, min_ovl=0.01):
    """Best column offset: right strip of a vs left strip of b."""
    H, W = a.shape
    best = (int(0.05 * W), -1)
    az = (a - a.mean()) / (a.std() + 1e-9)
    bz = (b - b.mean()) / (b.std() + 1e-9)
    for ovl in range(int(min_ovl * W), int(max_ovl * W)):
        sa = az[:, W - ovl:]
        sb = bz[:, :ovl]
        r = (sa * sb).mean()
        if r > best[1]:
            best = (ovl, r)
    return best


def build(tid):
    files = sorted(glob.glob(f'{DATA}/sem/SEM_{tid}/PlainImages/*.tif'),
                   key=lambda p: int(p.split('_')[-1].split('.')[0]))
    tiles = [np.asarray(Image.open(f).convert('L'), np.float32) for f in files]
    H, W = tiles[0].shape
    # subsample rows for speed
    ovls = []
    for k in range(len(tiles) - 1):
        o, r = seam_offset(tiles[k][::4], tiles[k + 1][::4])
        ovls.append((o, round(float(r), 3)))
    # build mosaic with per-seam offsets
    steps = [W - o for o, _ in ovls]
    total = sum(steps) + W
    mosaic = np.zeros((H, total), np.float32)
    wsum = np.zeros_like(mosaic)
    pos = 0
    for k, t in enumerate(tiles):
        mosaic[:, pos:pos + W] += t
        wsum[:, pos:pos + W] += 1
        if k < len(steps):
            pos += steps[k]
    mosaic /= np.maximum(wsum, 1)
    mosaic = np.fliplr(mosaic)
    px_mm = TILE_MM / W
    np.save(f'{CACHE}/mosaicX_{tid}.npy', mosaic[::4, ::8])
    print(f'T{tid}: overlaps px (corr): {ovls[:6]}... span={mosaic.shape[1]*px_mm:.1f}mm '
          f'(5%-assumption span was {(12*int(W*.95)+W)*px_mm:.1f}mm)')
    return mosaic, px_mm


def features(tid):
    mosaic, px_mm = build(tid)
    H = mosaic.shape[0]
    g = np.abs(np.diff(mosaic, axis=1))
    rv = np.convolve(g.mean(1), np.ones(51) / 51, 'same')
    tc = int(np.argmin(rv))
    bh = int(0.75 / px_mm)
    rows = np.r_[np.arange(0, max(0, tc - bh)), np.arange(min(H, tc + bh), H)]
    xcol = 100.0 - (mosaic.shape[1] - 1 - np.arange(mosaic.shape[1])) * px_mm
    edges = np.arange(20.0, 100.0 + 1e-9, 0.2)
    xc = 0.5 * (edges[:-1] + edges[1:])
    names = ['s_mean', 's_std', 's_grad', 's_locvar', 's_fftlow', 's_ffthigh', 's_entropy']
    out = {k: np.full(len(xc), np.nan) for k in names}
    for b, x0 in enumerate(xc):
        cm = (xcol >= x0 - 0.1) & (xcol < x0 + 0.1)
        if cm.sum() < 8:
            continue
        patch = mosaic[np.ix_(rows, np.flatnonzero(cm))]
        out['s_mean'][b] = patch.mean()
        out['s_std'][b] = patch.std()
        out['s_grad'][b] = np.abs(np.diff(patch, axis=0)).mean()
        mu = patch.mean(0)
        out['s_locvar'][b] = ((patch - mu) ** 2).mean()
        p = patch - patch.mean()
        F = np.abs(np.fft.rfft(p, axis=0)).mean(1)
        nn = len(F)
        out['s_fftlow'][b] = F[1:nn // 8].mean()
        out['s_ffthigh'][b] = F[nn // 2:].mean()
        h, _ = np.histogram(patch, bins=32, range=(0, 255), density=True)
        h = h[h > 0]
        out['s_entropy'][b] = -(h * np.log(h)).sum()
    out['x'] = xc
    np.savez(f'{CACHE}/semfeatX_{tid}.npz', **out)


if __name__ == '__main__':
    for t in [int(a) for a in sys.argv[1:]]:
        features(t)
