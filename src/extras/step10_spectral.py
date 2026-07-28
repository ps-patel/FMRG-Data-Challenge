"""Melt-pool temporal-dynamics features: sliding-window spectra of per-frame series.

At 50 fps / 10 mm/s: window of 32 frames = 0.64 s = 6.4 mm context, resolving
0-25 Hz melt-pool oscillations (ripple formation dynamics).
"""
import sys
import numpy as np

import fmrg_paths as _P
CACHE = str(_P.CACHE)
DATA = str(_P.DATA)
TRACKS = [8, 10, 14, 21]
SERIES = ['a1400', 'a1800', 't_max', 'cy1800', 'cx1800', 'wid1400', 'tail_len']
WIN = 32
FPS = 50.0


def spectral_feats(v, win=WIN):
    n = len(v)
    v = np.asarray(v, float)
    med = np.nanmedian(v)
    v = np.where(np.isfinite(v), v, med)
    half = win // 2
    freqs = np.fft.rfftfreq(win, d=1 / FPS)
    bands = {'b0_2': (0.1, 2), 'b2_8': (2, 8), 'b8_15': (8, 15), 'b15_25': (15, 25)}
    out = {f'{b}': np.full(n, np.nan) for b in bands}
    out['scent'] = np.full(n, np.nan)   # spectral centroid
    out['hfrac'] = np.full(n, np.nan)   # high-freq fraction
    out['rstd3'] = np.full(n, np.nan)   # short-scale roughness of series
    hann = np.hanning(win)
    for i in range(n):
        s = max(0, i - half)
        e = min(n, s + win)
        s = max(0, e - win)
        seg = v[s:e]
        if len(seg) < win:
            continue
        seg = seg - np.polyval(np.polyfit(np.arange(win), seg, 1), np.arange(win))
        F = np.abs(np.fft.rfft(seg * hann)) ** 2
        tot = F[1:].sum() + 1e-12
        for b, (lo, hi) in bands.items():
            m = (freqs >= lo) & (freqs < hi)
            out[b][i] = F[m].sum() / tot
        out['scent'][i] = (freqs[1:] * F[1:]).sum() / tot
        out['hfrac'][i] = F[freqs >= 10].sum() / tot
        out['rstd3'][i] = np.std(np.diff(v[max(0, i - 3):i + 4]))
    return out


def main():
    for tid in TRACKS:
        th = dict(np.load(f'{CACHE}/thfeat_{tid}.npz'))
        feats = {}
        for s in SERIES:
            sf = spectral_feats(th[s])
            for k, arr in sf.items():
                feats[f'sp_{s}_{k}'] = arr
        np.savez(f'{CACHE}/spfeat_{tid}.npz', **feats)
        print(f'track {tid}: {len(feats)} spectral features')


if __name__ == '__main__':
    main()
