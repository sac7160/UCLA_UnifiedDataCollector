"""
compare_imu_noise.py
────────────────────────────────────────────────────────────────────────────
Quantitatively compares the RAW noise level of watch IMU (imu.csv — a real
accelerometer/gyro chip) vs fingertip IMU (fingertip_imu.csv — accel/gyro
computed by differentiating camera-tracked position) for the same trials,
to test whether fingertip IMU's lower digit/letter-recognition accuracy
(vs watch IMU) traces back to it simply being noisier.

METHOD: for each channel, fit a smooth local trend (Savitzky-Golay) over a
window matched to a fixed TIME duration (not sample count — watch (~100Hz)
and fingertip (~20-30fps) have very different native sampling rates, so a
fixed sample-count window would smooth over very different amounts of real
time). The residual (raw - smoothed) is treated as "noise"; the reported
metric is noise_rms / smoothed_rms — a dimensionless ratio, comparable
across sources even though watch and fingertip accel/gyro are physically
different quantities in different units (this is why raw RMS values alone
would NOT be a fair comparison).

Runs on the ORIGINAL raw CSV samples, not the 64-step resampled sequences
the ML pipeline uses — that resampling already smooths things, so it isn't
representative of how noisy the underlying signal actually is.

Usage:
    python compare_imu_noise.py --dataset-root ../dataset --classes digits_0 digits_1 digits_2
    python compare_imu_noise.py --trial-dirs ../dataset/digits_0/trial_001 ../dataset/digits_0/trial_002
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

SMOOTH_WINDOW_SEC = 0.15   # matched real-time window for the smoothing filter, regardless of source's sample rate


def collect_trial_dirs(dataset_root: Path, classes: list) -> list:
    dirs = []
    for cls in classes:
        cls_dir = dataset_root / cls
        if not cls_dir.exists():
            print(f'[WARN] no folder for class "{cls}" under {dataset_root}')
            continue
        dirs.extend(sorted(p for p in cls_dir.glob('trial_*') if p.is_dir()))
    return dirs


def _savgol_window(dt: float) -> int:
    """Odd window length in samples covering SMOOTH_WINDOW_SEC of real
    time — this is what makes the comparison fair between two sources
    with very different native sample rates."""
    n = max(5, int(round(SMOOTH_WINDOW_SEC / dt)))
    return n if n % 2 == 1 else n + 1


def noise_ratio(t: np.ndarray, values: np.ndarray) -> dict | None:
    """values: (N, 3), one sensor's three axes. Returns per-channel and
    mean noise-to-signal ratio (residual RMS / smoothed RMS), plus the
    detected sample rate — or None if there's too little data to filter."""
    if len(t) < 8:
        return None
    order = np.argsort(t)
    t = t[order]; values = values[order]
    dt = np.median(np.diff(t))
    if not np.isfinite(dt) or dt <= 0:
        return None
    window = _savgol_window(dt)
    if len(values) < window + 2:
        return None

    ratios = []
    for c in range(3):
        smoothed = savgol_filter(values[:, c], window_length=window, polyorder=3)
        residual = values[:, c] - smoothed
        smoothed_rms = np.sqrt(np.mean(smoothed ** 2))
        residual_rms = np.sqrt(np.mean(residual ** 2))
        ratios.append(residual_rms / smoothed_rms if smoothed_rms > 1e-9 else np.nan)
    return {'per_channel': ratios, 'mean': float(np.nanmean(ratios)),
            'dt': float(dt), 'hz': float(1.0 / dt), 'n_samples': len(values)}


def load_watch_imu_noise(trial_dir: Path) -> dict:
    path = trial_dir / 'imu.csv'
    if not path.exists():
        return {'acc': None, 'gyro': None}
    df = pd.read_csv(path)
    acc = df[df['sensor'] == 'acc']
    gyro = df[df['sensor'] == 'gyro']
    return {
        'acc': noise_ratio(acc['time_aligned'].to_numpy(), acc[['v1', 'v2', 'v3']].to_numpy()),
        'gyro': noise_ratio(gyro['time_aligned'].to_numpy(), gyro[['v1', 'v2', 'v3']].to_numpy()),
    }


def load_fingertip_imu_noise(trial_dir: Path, finger: str) -> dict:
    path = trial_dir / 'fingertip_imu.csv'
    if not path.exists():
        return {'acc': None, 'gyro': None}
    df = pd.read_csv(path)
    df = df[(df['finger'] == finger) & (df['detected'] == 1)]
    if len(df) < 2:
        return {'acc': None, 'gyro': None}
    t = df['time_aligned'].to_numpy()
    return {
        'acc': noise_ratio(t, df[['accel_x', 'accel_y', 'accel_z']].to_numpy()),
        'gyro': noise_ratio(t, df[['gyro_x', 'gyro_y', 'gyro_z']].to_numpy()),
    }


def aggregate(trial_dirs: list, finger: str) -> dict:
    rows = {'watch_acc': [], 'watch_gyro': [], 'fingertip_acc': [], 'fingertip_gyro': []}
    hz = {'watch': [], 'fingertip': []}
    n_used = 0
    for trial_dir in trial_dirs:
        w = load_watch_imu_noise(trial_dir)
        f = load_fingertip_imu_noise(trial_dir, finger)
        used_this_trial = False
        if w['acc'] is not None:
            rows['watch_acc'].append(w['acc']['mean']); hz['watch'].append(w['acc']['hz']); used_this_trial = True
        if w['gyro'] is not None:
            rows['watch_gyro'].append(w['gyro']['mean'])
        if f['acc'] is not None:
            rows['fingertip_acc'].append(f['acc']['mean']); hz['fingertip'].append(f['acc']['hz']); used_this_trial = True
        if f['gyro'] is not None:
            rows['fingertip_gyro'].append(f['gyro']['mean'])
        if used_this_trial:
            n_used += 1
    print(f'[DATA] {n_used}/{len(trial_dirs)} trials had usable IMU data on at least one side')
    return {
        'n_trials': n_used,
        'watch_hz_median': float(np.median(hz['watch'])) if hz['watch'] else float('nan'),
        'fingertip_hz_median': float(np.median(hz['fingertip'])) if hz['fingertip'] else float('nan'),
        'stats': {k: {
            'mean': float(np.mean(v)) if v else float('nan'),
            'median': float(np.median(v)) if v else float('nan'),
            'std': float(np.std(v)) if v else float('nan'),
            'n': len(v),
        } for k, v in rows.items()},
        'raw': rows,
    }


def print_report(agg: dict, finger: str):
    print(f'\n=== noise-to-signal ratio (residual RMS / smoothed RMS), finger="{finger}" ===')
    print(f'median sample rate — watch: {agg["watch_hz_median"]:.1f} Hz   '
          f'fingertip: {agg["fingertip_hz_median"]:.1f} Hz')
    print(f'\n{"signal":18s} {"mean":>8s} {"median":>8s} {"std":>8s} {"n":>5s}')
    for key, s in agg['stats'].items():
        print(f'{key:18s} {s["mean"]:8.3f} {s["median"]:8.3f} {s["std"]:8.3f} {s["n"]:5d}')

    wa, fa = agg['stats']['watch_acc']['mean'], agg['stats']['fingertip_acc']['mean']
    wg, fg = agg['stats']['watch_gyro']['mean'], agg['stats']['fingertip_gyro']['mean']
    if wa and fa:
        print(f'\naccel: fingertip is {fa / wa:.2f}x noisier than watch' if fa > wa
              else f'\naccel: watch is {wa / fa:.2f}x noisier than fingertip')
    if wg and fg:
        print(f'gyro:  fingertip is {fg / wg:.2f}x noisier than watch' if fg > wg
              else f'gyro:  watch is {wg / fg:.2f}x noisier than fingertip')


def plot_summary(agg: dict, finger: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ['accel', 'gyro']
    watch_vals = [agg['stats']['watch_acc']['mean'], agg['stats']['watch_gyro']['mean']]
    watch_err = [agg['stats']['watch_acc']['std'], agg['stats']['watch_gyro']['std']]
    finger_vals = [agg['stats']['fingertip_acc']['mean'], agg['stats']['fingertip_gyro']['mean']]
    finger_err = [agg['stats']['fingertip_acc']['std'], agg['stats']['fingertip_gyro']['std']]

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, watch_vals, width, yerr=watch_err, capsize=4, label='watch IMU', color='#1f77b4')
    ax.bar(x + width / 2, finger_vals, width, yerr=finger_err, capsize=4, label='fingertip IMU', color='#d62728')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('noise-to-signal ratio (residual RMS / smoothed RMS)')
    ax.set_title(f'IMU noise level: watch vs. fingertip (finger="{finger}")')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_example_trial(trial_dir: Path, finger: str, save_path: Path):
    """Raw vs. smoothed overlay for one trial's accel-x channel, watch vs.
    fingertip side by side — makes the jitter difference visible directly,
    not just as a summary number."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

    df_w = pd.read_csv(trial_dir / 'imu.csv')
    acc_w = df_w[df_w['sensor'] == 'acc'].sort_values('time_aligned')
    t_w = acc_w['time_aligned'].to_numpy()
    v_w = acc_w['v1'].to_numpy()
    dt_w = np.median(np.diff(t_w))
    smoothed_w = savgol_filter(v_w, window_length=_savgol_window(dt_w), polyorder=3)
    axes[0].plot(t_w, v_w, color='#1f77b4', alpha=0.4, linewidth=1, label='raw')
    axes[0].plot(t_w, smoothed_w, color='#1f77b4', linewidth=2, label='smoothed')
    axes[0].set_title(f'Watch IMU accel (axis v1) — {1/dt_w:.0f} Hz')
    axes[0].set_ylabel('m/s\u00b2'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    df_f = pd.read_csv(trial_dir / 'fingertip_imu.csv')
    finger_df = df_f[(df_f['finger'] == finger) & (df_f['detected'] == 1)].sort_values('time_aligned')
    t_f = finger_df['time_aligned'].to_numpy()
    v_f = finger_df['accel_x'].to_numpy()
    dt_f = np.median(np.diff(t_f))
    smoothed_f = savgol_filter(v_f, window_length=_savgol_window(dt_f), polyorder=3)
    axes[1].plot(t_f, v_f, color='#d62728', alpha=0.4, linewidth=1, label='raw')
    axes[1].plot(t_f, smoothed_f, color='#d62728', linewidth=2, label='smoothed')
    axes[1].set_title(f'Fingertip IMU accel ({finger}, axis x) — {1/dt_f:.0f} Hz')
    axes[1].set_xlabel('time (s)'); axes[1].set_ylabel('a.u.'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.suptitle(f'Example trial: {trial_dir.name}', fontweight='bold')
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Compare watch IMU vs fingertip IMU noise level')
    parser.add_argument('--dataset-root', type=Path, default=None)
    parser.add_argument('--classes', nargs='+', default=None)
    parser.add_argument('--trial-dirs', nargs='+', type=Path, default=None,
                         help='explicit list of trial folders — alternative to --dataset-root/--classes')
    parser.add_argument('--finger', default='index')
    parser.add_argument('--out-dir', type=Path, default=Path('imu_noise_analysis'))
    args = parser.parse_args()

    if args.trial_dirs is not None:
        trial_dirs = args.trial_dirs
    elif args.dataset_root is not None and args.classes is not None:
        trial_dirs = collect_trial_dirs(args.dataset_root, args.classes)
    else:
        parser.error('pass either --trial-dirs, or both --dataset-root and --classes')

    if not trial_dirs:
        raise RuntimeError('no trial folders found')

    agg = aggregate(trial_dirs, args.finger)
    print_report(agg, args.finger)

    summary_path = args.out_dir / 'noise_comparison_summary.png'
    plot_summary(agg, args.finger, summary_path)
    print(f'\n[PLOT] summary bar chart: {summary_path}')

    example_path = args.out_dir / 'noise_comparison_example_trial.png'
    plot_example_trial(trial_dirs[0], args.finger, example_path)
    print(f'[PLOT] example trial overlay: {example_path}')

    with open(args.out_dir / 'noise_comparison_stats.json', 'w') as f:
        json.dump({k: v for k, v in agg.items() if k != 'raw'}, f, indent=2)
    print(f'[DATA] {args.out_dir}/noise_comparison_stats.json')


if __name__ == '__main__':
    main()