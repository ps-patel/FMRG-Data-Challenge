#!/usr/bin/env python3
"""One-command runner for the full FMRG pipeline (cross-platform, pure Python).

  python run_pipeline.py            # everything (ingest -> final model -> figures)
  python run_pipeline.py --from 4   # resume from step 4
  python run_pipeline.py --only 7   # run a single step

Steps:
  0 ingest raw data          1 extract labels        2 thermal features
  3 SEM features             4 build dataset         5 label ensemble
  6 z-profile                7 final model           9 figures
  (8 = optional XGBoost comparison: python src/s8_xgboost_compare.py)
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / 'src'
STEPS = {
    0: ('s0_ingest.py', []),
    1: ('s1_labels.py', []),
    2: ('s2_thermal_features.py', []),
    3: ('s3_sem_features.py', []),
    4: ('s4_dataset.py', []),
    5: ('s5_label_ensemble.py', []),
    6: ('s6_zprofile.py', []),
    7: ('s7_final_model.py', ['all']),
    9: ('s9_figures.py', []),
}


def run_step(n):
    script, args = STEPS[n]
    print(f'\n=== STEP {n}: {script} {" ".join(args)} ===')
    t0 = time.time()
    r = subprocess.run([sys.executable, str(SRC / script), *args], cwd=str(SRC))
    if r.returncode != 0:
        raise SystemExit(f'step {n} ({script}) failed with exit code {r.returncode}')
    print(f'=== STEP {n} done in {time.time()-t0:.0f}s ===')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='start', type=int, default=0)
    ap.add_argument('--only', type=int, default=None)
    a = ap.parse_args()
    todo = [a.only] if a.only is not None else [n for n in STEPS if n >= a.start]
    for n in todo:
        run_step(n)
    print('\nPipeline complete. Results: cache/final_results.json · figures/ folder')
