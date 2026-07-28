"""STEP 0 — Ingest raw data into fast numpy caches.

Thermal: parse MATLAB v5 file, detect the laser-on window, keep the final 400
frames (= 20-100 mm at 0.2 mm/frame), cache frames + per-frame x positions.
Height maps: parse Wyko ASCII, convert to mm, reorder x to physical direction,
crop to 20-100 mm, cache Z/x/y arrays.

Usage:  python s0_ingest.py [track_id ...]     (default: all four tracks)
"""
import sys
import numpy as np
from fmrg_paths import CACHE, THERMAL_DIR, HEIGHT_DIR, TRACKS, check_data
from fmrg_lib import (loadmat_v5, find_thermal_array, detect_laser_interval,
                      load_wyko_asc, N_FRAMES, MM_PER_FRAME, COMMON_X_END_MM)


def ingest_thermal(tid):
    mat = loadmat_v5(str(THERMAL_DIR / f'Thermal_{tid}.mat'))
    frames, key = find_thermal_array(mat)
    score = np.percentile(frames.reshape(frames.shape[0], -1), 99.5, axis=1)
    on_start, on_stop, thr = detect_laser_interval(score)
    stop = int(on_stop)
    start = max(0, stop - N_FRAMES)
    seg = frames[start:stop]
    idx = np.arange(start, stop)
    x_mm = COMMON_X_END_MM - ((stop - idx) - 0.5) * MM_PER_FRAME
    np.save(CACHE / f'th_seg_{tid}.npy', seg)
    np.save(CACHE / f'th_x_{tid}.npy', x_mm)
    print(f'  thermal T{tid}: raw {frames.shape[0]} frames, laser on '
          f'[{on_start},{on_stop}), extracted [{start},{stop}) -> {seg.shape}')


def ingest_heightmap(tid):
    hm = load_wyko_asc(str(HEIGHT_DIR / f'Heightmap_{tid}.ASC'))
    np.save(CACHE / f'hm_Z_{tid}.npy', hm['Z_mm'])
    np.save(CACHE / f'hm_x_{tid}.npy', hm['x_mm'])
    np.save(CACHE / f'hm_y_{tid}.npy', hm['y_mm'])
    Z = hm['Z_mm']
    print(f'  heightmap T{tid}: {Z.shape}, NaN {np.isnan(Z).mean():.3f}')


if __name__ == '__main__':
    check_data()
    tracks = [int(a) for a in sys.argv[1:]] or TRACKS
    for t in tracks:
        print(f'[ingest] track {t}')
        ingest_thermal(t)
        ingest_heightmap(t)
