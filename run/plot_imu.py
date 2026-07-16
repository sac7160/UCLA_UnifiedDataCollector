"""
plot_imu_comparison.py
────────────────────────────────────────────────────────────────────────────
Plots watch IMU (imu.csv) and fingertip virtual IMU (fingertip_imu.csv) on a
shared time axis, so the two sources can be visually cross-checked for
capture quality and rough alignment.

Usage:
    python plot_imu_comparison.py --imu imu.csv --fingertip fingertip_imu.csv
    python plot_imu_comparison.py --imu imu.csv --fingertip fingertip_imu.csv --finger thumb
    python plot_imu_comparison.py --imu imu.csv --fingertip fingertip_imu.csv --save out.png
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def load_watch_imu(path: str):
    """imu.csv columns: time_aligned, sensor, v1, v2, v3
    sensor is 'acc' or 'gyro'."""
    df = pd.read_csv(path)
    acc  = df[df['sensor'] == 'acc'].sort_values('time_aligned')
    gyro = df[df['sensor'] == 'gyro'].sort_values('time_aligned')
    return acc, gyro


def load_fingertip_imu(path: str, finger: str):
    """fingertip_imu.csv columns: time_aligned, finger, hand_label, detected,
    accel_x/y/z, gyro_x/y/z, pos_x/y/z."""
    df = pd.read_csv(path)
    df = df[df['finger'] == finger].sort_values('time_aligned')
    if df['detected'].eq(0).any():
        n_missing = int((df['detected'] == 0).sum())
        print(f'[warn] {n_missing}/{len(df)} "{finger}" frames have detected=0 (hand not found)')
    return df


def plot_comparison(watch_acc, watch_gyro, finger_df, finger_name: str, save_path: str | None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex='col')

    axis_labels  = ['x', 'y', 'z']
    axis_colors  = ['#d62728', '#2ca02c', '#1f77b4']

    # Watch accel (top-left)
    ax = axes[0, 0]
    for col, label, color in zip(['v1', 'v2', 'v3'], axis_labels, axis_colors):
        ax.plot(watch_acc['time_aligned'], watch_acc[col], label=label, color=color, linewidth=0.9)
    ax.set_title('Watch IMU — accel')
    ax.set_ylabel('accel')
    ax.legend(loc='upper right', fontsize=8)

    # Watch gyro (bottom-left)
    ax = axes[1, 0]
    for col, label, color in zip(['v1', 'v2', 'v3'], axis_labels, axis_colors):
        ax.plot(watch_gyro['time_aligned'], watch_gyro[col], label=label, color=color, linewidth=0.9)
    ax.set_title('Watch IMU — gyro')
    ax.set_ylabel('gyro')
    ax.set_xlabel('time (s)')
    ax.legend(loc='upper right', fontsize=8)

    # Fingertip accel (top-right)
    ax = axes[0, 1]
    for col, label, color in zip(['accel_x', 'accel_y', 'accel_z'], axis_labels, axis_colors):
        ax.plot(finger_df['time_aligned'], finger_df[col], label=label, color=color, linewidth=0.9)
    ax.set_title(f'Fingertip virtual IMU ({finger_name}) — accel')
    ax.legend(loc='upper right', fontsize=8)

    # Fingertip gyro (bottom-right)
    ax = axes[1, 1]
    for col, label, color in zip(['gyro_x', 'gyro_y', 'gyro_z'], axis_labels, axis_colors):
        ax.plot(finger_df['time_aligned'], finger_df[col], label=label, color=color, linewidth=0.9)
    ax.set_title(f'Fingertip virtual IMU ({finger_name}) — gyro')
    ax.set_xlabel('time (s)')
    ax.legend(loc='upper right', fontsize=8)

    fig.suptitle('Watch IMU vs. Fingertip Virtual IMU', fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f'[saved] {save_path}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot watch IMU and fingertip virtual IMU together')
    parser.add_argument('--imu', required=True, help='Path to imu.csv (watch IMU)')
    parser.add_argument('--fingertip', required=True, help='Path to fingertip_imu.csv')
    parser.add_argument('--finger', default='index',
                         choices=['thumb', 'index', 'middle', 'ring', 'pinky'],
                         help='Which finger to plot from fingertip_imu.csv (default: index)')
    parser.add_argument('--save', default=None, help='Save the figure to this path instead of showing it')
    args = parser.parse_args()

    watch_acc, watch_gyro = load_watch_imu(args.imu)
    finger_df = load_fingertip_imu(args.fingertip, args.finger)

    plot_comparison(watch_acc, watch_gyro, finger_df, args.finger, args.save)


if __name__ == '__main__':
    main()