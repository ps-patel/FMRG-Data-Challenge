"""STEP 3 — SEM substrate-texture features (organizer-official alignment).

Stitch tiles 01..N left-to-right with 5% overlap (no per-tile flips), then
fliplr the whole mosaic -> same direction as the height map (x=100 on right).
Mask the track band +/-0.75 mm (challenge leakage rule); compute texture
features from the substrate flanks only.

Writes semfeat_{tid}.npz + mosaic_{tid}.npy (downsampled, for figures).

Usage:  python s3_sem_features.py [track_id ...]
"""
import sys
import numpy as np
from PIL import Image
from fmrg_paths import CACHE, SEM_DIR, TRACKS

TILE_MM = 6.41
OVL = 0.05
MASK_HALF_MM = 0.75


def tile_files(tid):
    root = SEM_DIR / f'SEM_{tid}' / 'PlainImages'
    files = sorted(root.glob('*.tif'),
                   key=lambda p: int(p.stem.split('_')[-1]))
    if not files:
        raise SystemExit(f'no SEM tiles found under {root}')
    return files


def build_mosaic(tid):
    tiles = [np.asarray(Image.open(f).convert('L'), np.float32)
             for f in tile_files(tid)]
    H, W = tiles[0].shape
    step = int(round(W * (1 - OVL)))
    mosaic = np.zeros((H, step * (len(tiles) - 1) + W), np.float32)
    wsum = np.zeros_like(mosaic)
    for k, t in enumerate(tiles):
        mosaic[:, k * step:k * step + W] += t
        wsum[:, k * step:k * step + W] += 1
    mosaic /= np.maximum(wsum, 1)
    mosaic = np.fliplr(mosaic)                  # organizer convention
    px_mm = TILE_MM / W
    return mosaic, px_mm


def sem_features(tid):
    mosaic, px_mm = build_mosaic(tid)
    H = mosaic.shape[0]
    g = np.abs(np.diff(mosaic, axis=1))
    rv = np.convolve(g.mean(1), np.ones(51) / 51, 'same')
    tc = int(np.argmin(rv))                     # smooth melt band row
    bh = int(MASK_HALF_MM / px_mm)
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
    np.savez(CACHE / f'semfeat_{tid}.npz', **out)
    np.save(CACHE / f'mosaic_{tid}.npy', mosaic[::4, ::8])
    ok = int(np.isfinite(out['s_mean']).sum())
    print(f'  sem features T{tid}: mosaic span {mosaic.shape[1]*px_mm:.1f} mm, '
          f'track row {tc}, {ok}/400 bins')


if __name__ == '__main__':
    for t in [int(a) for a in sys.argv[1:]] or TRACKS:
        sem_features(t)
