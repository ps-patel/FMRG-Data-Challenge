"""Central path configuration — everything is relative to the repo root by default.

Override with environment variables if your data lives elsewhere:
  FMRG_DATA    folder containing thermal/, height_maps/, sem/   (default: <repo>/data)
  FMRG_CACHE   working cache for intermediate arrays/tables     (default: <repo>/cache)
  FMRG_FIGURES output figures                                   (default: <repo>/figures)
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get('FMRG_ROOT', str(Path(__file__).resolve().parents[1])))
DATA = Path(os.environ.get('FMRG_DATA', str(ROOT / 'data')))
CACHE = Path(os.environ.get('FMRG_CACHE', str(ROOT / 'cache')))
FIGURES = Path(os.environ.get('FMRG_FIGURES', str(ROOT / 'figures')))

for _p in (CACHE, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

THERMAL_DIR = DATA / 'thermal'
HEIGHT_DIR = DATA / 'height_maps'
SEM_DIR = DATA / 'sem'

TRACKS = [8, 10, 14, 21]
POWER = {8: 400, 10: 350, 14: 300, 21: 200}   # confirmed by challenge organizers
TRAIN_TRACKS = [8, 10, 14]
TEST_TRACK = 21


def check_data():
    missing = []
    for tid in TRACKS:
        if not (THERMAL_DIR / f'Thermal_{tid}.mat').exists():
            missing.append(f'thermal/Thermal_{tid}.mat')
        if not (HEIGHT_DIR / f'Heightmap_{tid}.ASC').exists():
            missing.append(f'height_maps/Heightmap_{tid}.ASC')
        if not (SEM_DIR / f'SEM_{tid}' / 'PlainImages').exists():
            missing.append(f'sem/SEM_{tid}/PlainImages/')
    if missing:
        raise SystemExit(
            'Missing raw data under ' + str(DATA) + ':\n  ' + '\n  '.join(missing) +
            '\nDownload the three zips from https://doi.org/10.5281/zenodo.21285367 '
            'and extract them into the data/ folder (see README).')
    return True
