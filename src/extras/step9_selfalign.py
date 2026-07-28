"""Per-track self-alignment experiment.

(A) ORACLE: per-track lag maximizing corr(thermal melt width, label width).
    Uses labels -> diagnostic ceiling only.
(B) LABEL-FREE: extract the track's apparent width profile from SEM tiles
    (track region used for REGISTRATION only, never as a model feature),
    cross-correlate with the thermal melt-pool width profile -> per-track lag.
"""
import sys, glob, json
import numpy as np
import pandas as pd
from PIL import Image

import fmrg_paths as _P
CACHE = str(_P.CACHE)
DATA = str(_P.DATA)
TRACKS = [8, 10, 14, 21]
TILE_MM = 6.41


def sem_track_width_profile(tid):
    """Per 0.2mm bin: width of the smooth (melted) band in SEM tiles."""
    files = sorted(glob.glob(f'{DATA}/sem/SEM_{tid}/PlainImages/*.tif'),
                   key=lambda p: int(p.split('_')[-1].split('.')[0]))
    edges = np.arange(20.0, 100.0 + 1e-9, 0.2)
    xc = 0.5 * (edges[:-1] + edges[1:])
    prof = np.full(len(xc), np.nan)
    tiles = [np.asarray(Image.open(f).convert('L'), np.float32) for f in files]
    H, W = tiles[0].shape
    px_mm = TILE_MM / W
    px_mm_y = px_mm  # assume square pixels
    for k, im in enumerate(tiles):
        # smoothness map: local horizontal gradient magnitude, box-smoothed
        g = np.abs(np.diff(im, axis=1))
        g = np.c_[g, g[:, -1:]]
        # column-block processing at 0.2mm resolution
        ncol_bin = max(1, int(0.2 / px_mm))
        # track center for tile
        rv = np.convolve(g.mean(1), np.ones(51) / 51, 'same')
        c = int(np.argmin(rv))
        thr = np.percentile(rv, 20) * 1.6  # smooth threshold
        for cb in range(0, W - ncol_bin + 1, ncol_bin):
            colg = g[:, cb:cb + ncol_bin].mean(1)
            colg = np.convolve(colg, np.ones(31) / 31, 'same')
            m = colg < thr
            # contiguous run through c
            if not m[c]:
                continue
            lo = c
            while lo > 0 and m[lo - 1]:
                lo -= 1
            hi = c
            while hi < H - 1 and m[hi + 1]:
                hi += 1
            wmm = (hi - lo + 1) * px_mm_y
            # map to physical x: tile k covers [100-(k+1)*T, 100-k*T]; col 0 at 100-k*T side
            x_phys = 100.0 - k * TILE_MM - (cb + ncol_bin / 2) * px_mm
            b = int((x_phys - 20.0) / 0.2)
            if 0 <= b < len(xc):
                prof[b] = wmm
    return xc, prof


def best_lag(a, b, max_lag=20):
    best = (0, 0.0)
    for lag in range(-max_lag, max_lag + 1):
        aa = pd.Series(a).shift(lag).values
        m = np.isfinite(aa) & np.isfinite(b)
        if m.sum() < 80:
            continue
        x = aa[m] - aa[m].mean()
        y = b[m] - b[m].mean()
        if x.std() < 1e-12 or y.std() < 1e-12:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if abs(r) > abs(best[1]):
            best = (lag, r)
    return best


def main():
    D = pd.read_csv(f'{CACHE}/datasetW.csv')
    out = {}
    for tid in TRACKS:
        d = D[D['track'] == tid]
        th = d['th_wid1400_m5'].values
        lab = d['width'].values
        # (A) oracle
        lagA, rA = best_lag(th, lab)
        # (B) label-free vs SEM band width
        xc, sem_prof = sem_track_width_profile(tid)
        np.save(f'{CACHE}/semband_{tid}.npy', sem_prof)
        lagB, rB = best_lag(th, sem_prof * 1e3)
        # also: does SEM band width itself predict label width locally?
        _, rSem = best_lag(sem_prof * 1e3, lab, max_lag=5)
        out[tid] = dict(oracle_lag=lagA, oracle_r=round(rA, 3),
                        semreg_lag=lagB, semreg_r=round(rB, 3),
                        sem_vs_label_r=round(rSem, 3))
        print(f'T{tid}: ORACLE lag={lagA:+d} r={rA:+.2f} | SEM-REG lag={lagB:+d} '
              f'r={rB:+.2f} | SEM-band vs label r={rSem:+.2f}')
    json.dump(out, open(f'{CACHE}/selfalign.json', 'w'))


if __name__ == '__main__':
    main()
