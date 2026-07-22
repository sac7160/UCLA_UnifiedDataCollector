"""
plot_trajectory.py
────────────────────────────────────────────────────────────────────────────
Plots the index-fingertip trajectory recorded to trajectory.csv (see
trajectory_calibration.py for column definitions) — either a full session's
file (data/session_.../trajectory.csv) or a single trial's cropped copy
(dataset/<label>/trial_NNN/trajectory.csv). Shows the 2D path — mic-anchored
mm once calibrated, else normalized image-plane coordinates — colored by
time, plus x/y and height vs. time underneath.

Usage:
    python plot_trajectory.py --trajectory data/session_20260722_105556/trajectory.csv
    python plot_trajectory.py --trajectory dataset/unlabeled/trial_002/trajectory.csv
    python plot_trajectory.py --trajectory trajectory.csv --save out.png
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def load_trajectory(path: str) -> pd.DataFrame:
    """trajectory.csv's first column is 'timestamp_sec' (session-level file)
    or 'time_aligned' (trial-level file, 0 = trial start) — same time base
    either way, just relative to a different zero point, so it's read
    generically here rather than assuming one name. Remaining columns:
    detected, x_px, y_px, x_norm, y_norm, pos_x/y/z_mm, local_x/y/z_mm,
    calibrated, global_x/y_mm, height_mm, frame_x/y_mm."""
    df = pd.read_csv(path)
    t_col = df.columns[0]
    df = df.rename(columns={t_col: 't'}).sort_values('t')
    n_missing = int((df['detected'] == 0).sum())
    if n_missing:
        print(f'[warn] {n_missing}/{len(df)} frames have detected=0 (hand not found)')
    return df[df['detected'] == 1].copy()


def plot_trajectory(df: pd.DataFrame, save_path: str | None):
    has_global = bool(df['calibrated'].any()) and df['global_x_mm'].notna().any()
    if has_global:
        x, y = df['global_x_mm'], df['global_y_mm']
        xlabel, ylabel = 'x (mm, mic-anchored)', 'y (mm, mic-anchored)'
    else:
        x, y = df['x_norm'], df['y_norm']
        xlabel, ylabel = 'x (normalized image-plane)', 'y (normalized image-plane)'

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])

    ax_path = fig.add_subplot(gs[:, 0])
    ax_path.plot(x, y, color='#888', linewidth=0.5, alpha=0.5, zorder=0)
    sc = ax_path.scatter(x, y, c=df['t'], cmap='viridis', s=6, zorder=1)
    ax_path.set_xlabel(xlabel)
    ax_path.set_ylabel(ylabel)
    ax_path.set_title('Index fingertip path' + (' (calibrated)' if has_global else ' (uncalibrated)'))
    ax_path.set_aspect('equal', adjustable='datalim')
    ax_path.invert_yaxis()   # image/mic-plane convention: y increases downward
    fig.colorbar(sc, ax=ax_path, label='time (s)')

    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xy.plot(df['t'], x, label='x', color='#d62728', linewidth=0.9)
    ax_xy.plot(df['t'], y, label='y', color='#2ca02c', linewidth=0.9)
    ax_xy.set_ylabel('position')
    ax_xy.set_title('x / y vs. time')
    ax_xy.legend(loc='upper right', fontsize=8)

    ax_h = fig.add_subplot(gs[1, 1])
    if df['height_mm'].notna().any():
        ax_h.plot(df['t'], df['height_mm'], color='#1f77b4', linewidth=0.9)
        ax_h.set_ylabel('height (mm)')
    else:
        ax_h.text(0.5, 0.5, 'height_mm not available\n(no camera-to-surface distance calibrated)',
                   ha='center', va='center', transform=ax_h.transAxes, fontsize=9, color='#888')
    ax_h.set_xlabel('time (s)')
    ax_h.set_title('height above surface')

    fig.suptitle('Index Fingertip Trajectory', fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'[saved] {save_path}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot an index-fingertip trajectory.csv')
    parser.add_argument('--trajectory', required=True, help='Path to trajectory.csv (session- or trial-level)')
    parser.add_argument('--save', default=None, help='Save the figure to this path instead of showing it')
    args = parser.parse_args()

    df = load_trajectory(args.trajectory)
    if df.empty:
        print('[warn] no detected frames in this file — nothing to plot')
        return
    plot_trajectory(df, args.save)


if __name__ == '__main__':
    main()
