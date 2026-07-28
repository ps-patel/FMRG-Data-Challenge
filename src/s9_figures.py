"""STEP 9 — Key figures.

Usage:  python s9_figures.py            # all
        python s9_figures.py scoremaps final ...
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fmrg_paths import CACHE, FIGURES, TRACKS, POWER

plt.rcParams.update({'figure.dpi': 110, 'font.size': 9})


def scoremaps():
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for ax, tid in zip(axes, TRACKS):
        S = np.load(CACHE / f'hm_score_{tid}.npy')
        lab = np.load(CACHE / f'labels_{tid}.npz')
        x = lab['col_x'][::10]
        y = np.arange(S.shape[0]) * 0.003982
        ax.imshow(S, extent=[x.min(), x.max(), y[-1], y[0]], aspect='auto', cmap='viridis')
        ax.plot(lab['x'], lab['yl'], 'r-', lw=0.7)
        ax.plot(lab['x'], lab['yr'], 'r-', lw=0.7)
        ax.set_ylabel(f'T{tid} ({POWER[tid]}W)\ny [mm]')
    axes[-1].set_xlabel('scan position x [mm]')
    axes[0].set_title('Melt-zone score maps with extracted track boundaries (red)')
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig_scoremaps.png', bbox_inches='tight')
    plt.close()


def width_power():
    D = pd.read_csv(CACHE / 'dataset.csv')
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for tid in TRACKS:
        d = D[D['track'] == tid]
        axes[0].plot(d['x'], d['width'], lw=1, label=f'T{tid} ({POWER[tid]}W)')
    axes[0].set_xlabel('x [mm]')
    axes[0].set_ylabel('local width [um]')
    axes[0].legend(ncol=4)
    axes[0].set_title('Extracted local track width vs position')
    g = D.groupby('power')['width'].median()
    axes[1].plot(g.index, g.values, 'o-')
    axes[1].set_xlabel('laser power [W]')
    axes[1].set_ylabel('median width [um]')
    axes[1].set_title('Power scaling (note nonlinearity)')
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig_width_power.png', bbox_inches='tight')
    plt.close()


def final():
    d = np.load(CACHE / 'final_test21.npz')
    x, y, p, pc, Q = d['x'], d['y'], d['p'], d['pc'], d['Q']
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax = axes[0]
    ax.fill_between(x, Q[0], Q[4], alpha=.18, label='90% PI (conformal)')
    ax.fill_between(x, Q[1], Q[3], alpha=.32, label='50% PI')
    ax.plot(x, p, 'r-', lw=1.2, label='predicted width')
    ax.plot(x, y, 'k.', ms=2.5, label='measured width')
    ax.set_ylabel('width [um]')
    ax.legend(ncol=4, fontsize=8)
    ax.set_title('FINAL model - Track 21 holdout (200 W)')
    ax = axes[1]
    ax.plot(x, d['yl_true'], 'k.', ms=2.5)
    ax.plot(x, d['yr_true'], 'k.', ms=2.5)
    ax.plot(x, pc - p / 2, 'r-', lw=1.1, label='predicted left boundary')
    ax.plot(x, pc + p / 2, 'b-', lw=1.1, label='predicted right boundary')
    ax.plot(x, pc, 'g--', lw=.9, label='predicted centerline')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [um]')
    ax.legend(ncol=3, fontsize=8)
    ax.set_title('Boundary representation y_left(x), y_right(x)')
    plt.tight_layout()
    plt.savefig(FIGURES / 'fig_final_test21.png', bbox_inches='tight')
    plt.close()


ALL = {'scoremaps': scoremaps, 'width_power': width_power, 'final': final}

if __name__ == '__main__':
    names = sys.argv[1:] or list(ALL)
    for n in names:
        ALL[n]()
        print(f'  figure: {n} -> {FIGURES}')
