"""
analyze_finger_motion.py
────────────────────────────────────────────────────────────────────────────
Quantifies fingertip motion/articulation from fingertip_imu.csv across a
whole dataset (all trials, all classes) — built for comparing collection
conditions (e.g. rigid vs. flexible finger writing), but works standalone
on just one condition too.

WHAT'S ACTUALLY IN THE DATA: fingertip_imu.csv has, per frame, each of 5
fingers' TIP position (pos_x/y/z, mm, MediaPipe world-landmark frame) plus
accel/gyro (in the tip's own local coordinate frame — see
fingertip_imu_multi.py). There's no wrist or MCP/PIP/DIP landmark saved.
Every metric below is built from tip position + accel/gyro across the 5
fingers, using two different position references to separate "the whole
hand moved" from "the fingers moved relative to each other":

  - per-trial mean     each finger's own average position over that
                         trial — spread here still includes genuine
                         hand/wrist translation across the trial.
  - per-frame hand
    centroid            at every single frame, subtract the average
                         position across all 5 fingertips *at that frame*
                         — cancels out whatever the whole hand is doing
                         and isolates how spread out the fingers are
                         relative to each other at each instant. Closer
                         to "articulation" in the relative-finger-motion
                         sense.

Plus, independent of either reference point:
  - accel / gyro RMS         magnitude of actual sensed acceleration/
                               rotation at each fingertip.
  - path length per second    total tip-to-tip travel distance, so a
                               finger that moves a lot but stays within a
                               small region (e.g. tiny fast wiggles) isn't
                               invisible to spread-based metrics alone.
  - pairwise distance std     how much the distance between each pair of
                               fingertips changes over a trial. Near-zero
                               if the whole hand moves as one rigid unit;
                               larger the more independently the fingers
                               move relative to each other.

Usage:
    # one condition on its own — saves stats.json + positions.npz for
    # later comparison, and a standalone figure for this condition alone
    python analyze_finger_motion.py --dataset-root dataset --out-dir analysis/flexible --label flexible

    # once you have both conditions analyzed:
    python analyze_finger_motion.py --compare analysis/flexible/stats.json analysis/rigid/stats.json \\
        --compare-out analysis/rigid_vs_flexible.png
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky']
FINGER_COLORS = {
    'thumb':  '#7f7f7f',
    'index':  '#1f77b4',
    'middle': '#ff7f0e',
    'ring':   '#2ca02c',
    'pinky':  '#9467bd',
}
POS_COLS = ['pos_x', 'pos_y', 'pos_z']
ACCEL_COLS = ['accel_x', 'accel_y', 'accel_z']
GYRO_COLS = ['gyro_x', 'gyro_y', 'gyro_z']


# ─── Scanning ──────────────────────────────────────────────────────────────────
def collect_trial_dirs(dataset_root: Path, classes: list) -> list:
    dirs = []
    for cls in classes:
        cls_dir = dataset_root / cls
        if not cls_dir.exists():
            print(f'[WARN] no folder for class "{cls}" under {dataset_root}')
            continue
        dirs.extend(sorted(p for p in cls_dir.glob('trial_*') if p.is_dir()))
    return dirs


# ─── Per-trial metric extraction ───────────────────────────────────────────────
def _wide_positions(df: pd.DataFrame) -> pd.DataFrame | None:
    """Pivots the long (one row per finger per frame) CSV into a wide
    table indexed by time_aligned, with a (finger, xyz) column per
    fingertip — needed for the per-frame cross-finger centroid, which by
    definition needs all 5 fingers' positions at the same instant side by
    side. Returns None if any finger is missing entirely from this trial."""
    if not set(FINGERS).issubset(set(df['finger'].unique())):
        return None
    pivoted = df.pivot_table(index='time_aligned', columns='finger', values=POS_COLS)
    return pivoted.sort_index()


def extract_trial_metrics(trial_dir: Path) -> dict | None:
    csv_path = trial_dir / 'fingertip_imu.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df[df['detected'] == 1]
    if df.empty:
        return None

    wide = _wide_positions(df)

    out = {
        'trial_mean_rel':   {f: None for f in FINGERS},   # (N,3) pos - this finger's own trial mean
        'frame_centroid_rel': {f: None for f in FINGERS},  # (N,3) pos - that frame's 5-finger centroid
        'frame_centroid_variability': {f: None for f in FINGERS},  # frame_centroid_rel with its own
                                                                     # trial mean removed — see below
        'accel': {f: None for f in FINGERS},
        'gyro': {f: None for f in FINGERS},
        'path_length_mm': {f: None for f in FINGERS},
        'duration_sec': float(df['time_aligned'].max() - df['time_aligned'].min()),
    }

    for finger in FINGERS:
        fdf = df[df['finger'] == finger].sort_values('time_aligned')
        if len(fdf) < 2:
            continue
        pos = fdf[POS_COLS].to_numpy()
        out['trial_mean_rel'][finger] = pos - pos.mean(axis=0, keepdims=True)
        out['accel'][finger] = fdf[ACCEL_COLS].to_numpy()
        out['gyro'][finger] = fdf[GYRO_COLS].to_numpy()
        out['path_length_mm'][finger] = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))

    if wide is not None:
        for finger in FINGERS:
            finger_pos = wide.xs(finger, axis=1, level='finger').to_numpy()   # (N, 3), columns pos_x/y/z
            centroid = np.stack([wide.xs(f, axis=1, level='finger').to_numpy() for f in FINGERS], axis=0).mean(axis=0)
            rel = finger_pos - centroid
            out['frame_centroid_rel'][finger] = rel   # raw — keeps each finger's anatomical offset from
                                                        # the centroid, which is what makes the scatter plot
                                                        # visually separate the fingers by shape/position
            out['frame_centroid_variability'][finger] = rel - rel.mean(axis=0, keepdims=True)   # mean-removed —
                                                        # this trial's anatomical offset subtracted back out,
                                                        # leaving only how much that offset *varied* — i.e.
                                                        # actual motion, not "where the finger normally sits"

        # Pairwise fingertip distance at each frame, and its std over the trial —
        # near-constant distance = the hand moved as one rigid unit.
        pair_std = {}
        for f1, f2 in combinations(FINGERS, 2):
            p1 = wide.xs(f1, axis=1, level='finger').to_numpy()
            p2 = wide.xs(f2, axis=1, level='finger').to_numpy()
            dist = np.linalg.norm(p1 - p2, axis=1)
            pair_std[f'{f1}-{f2}'] = float(np.std(dist))
        out['pairwise_distance_std'] = pair_std
    else:
        out['pairwise_distance_std'] = None

    return out


# ─── Aggregation across a whole dataset (condition) ────────────────────────────
def aggregate_condition(dataset_root: Path, classes: list) -> dict:
    trial_dirs = collect_trial_dirs(dataset_root, classes)
    trial_metrics = []
    for trial_dir in trial_dirs:
        m = extract_trial_metrics(trial_dir)
        if m is not None:
            trial_metrics.append(m)
    print(f'[DATA] {len(trial_metrics)}/{len(trial_dirs)} trials had usable fingertip_imu.csv')
    if not trial_metrics:
        raise RuntimeError(f'No usable fingertip_imu.csv found under {dataset_root} for classes {classes}')

    agg = {'n_trials': len(trial_metrics), 'stats': {}, 'raw_positions': {}}

    for finger in FINGERS:
        trial_mean_rel = np.concatenate(
            [m['trial_mean_rel'][finger] for m in trial_metrics if m['trial_mean_rel'][finger] is not None], axis=0)
        frame_centroid_chunks = [m['frame_centroid_rel'][finger] for m in trial_metrics
                                  if m.get('frame_centroid_rel', {}).get(finger) is not None]
        frame_centroid_rel = (np.concatenate(frame_centroid_chunks, axis=0)
                              if frame_centroid_chunks else np.zeros((0, 3)))
        variability_chunks = [m['frame_centroid_variability'][finger] for m in trial_metrics
                               if m.get('frame_centroid_variability', {}).get(finger) is not None]
        frame_centroid_variability = (np.concatenate(variability_chunks, axis=0)
                                      if variability_chunks else np.zeros((0, 3)))
        accel = np.concatenate([m['accel'][finger] for m in trial_metrics if m['accel'][finger] is not None], axis=0)
        gyro = np.concatenate([m['gyro'][finger] for m in trial_metrics if m['gyro'][finger] is not None], axis=0)
        path_lengths = [m['path_length_mm'][finger] for m in trial_metrics if m['path_length_mm'][finger] is not None]
        durations = [m['duration_sec'] for m in trial_metrics if m['path_length_mm'][finger] is not None]
        path_per_sec = [pl / d for pl, d in zip(path_lengths, durations) if d > 0]

        agg['raw_positions'][finger] = {
            'trial_mean_rel': trial_mean_rel.tolist(),
            'frame_centroid_rel': frame_centroid_rel.tolist(),
            'frame_centroid_variability': frame_centroid_variability.tolist(),
        }
        agg['stats'][finger] = {
            'pos_spread_trial_mean_rms_mm': float(np.sqrt(np.mean(np.sum(trial_mean_rel[:, :2] ** 2, axis=1)))),
            'pos_spread_frame_centroid_rms_mm': (
                # RMS around *this finger's own mean* in the frame-centroid frame — not RMS from zero.
                # RMS-from-zero would also count each finger's anatomical distance from the hand centroid
                # (e.g. thumb sits much farther from centroid than index does, regardless of how much
                # either one actually moved), which swamped the real motion signal this metric is meant
                # to capture. See extract_trial_metrics()'s frame_centroid_variability.
                float(np.sqrt(np.mean(np.sum(frame_centroid_variability[:, :2] ** 2, axis=1))))
                if len(frame_centroid_variability) else float('nan')),
            'accel_rms_mm_s2': float(np.sqrt(np.mean(np.sum(accel ** 2, axis=1)))),
            'gyro_rms_rad_s': float(np.sqrt(np.mean(np.sum(gyro ** 2, axis=1)))),
            'path_length_mm_per_s': float(np.mean(path_per_sec)) if path_per_sec else float('nan'),
        }

    pair_stds = {}
    for f1, f2 in combinations(FINGERS, 2):
        vals = [m['pairwise_distance_std'][f'{f1}-{f2}'] for m in trial_metrics
                if m.get('pairwise_distance_std') is not None]
        if vals:
            pair_stds[f'{f1}-{f2}'] = float(np.mean(vals))
    agg['pairwise_distance_std_mm'] = pair_stds

    return agg


def save_condition(agg: dict, out_dir: Path, label: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_payload = {'label': label, 'n_trials': agg['n_trials'],
                      'stats': agg['stats'], 'pairwise_distance_std_mm': agg['pairwise_distance_std_mm']}
    with open(out_dir / 'stats.json', 'w') as f:
        json.dump(stats_payload, f, indent=2)
    np.savez(out_dir / 'positions.npz', **{
        f'{finger}__{ref}': np.array(agg['raw_positions'][finger][ref])
        for finger in FINGERS for ref in ('trial_mean_rel', 'frame_centroid_rel', 'frame_centroid_variability')
    })


def load_condition(stats_path: Path) -> dict:
    """Loads stats.json and, if present alongside it, positions.npz (saved
    separately from the JSON since per-frame position arrays are the one
    thing large enough that dumping them straight into JSON would make it
    unwieldy) — merges both into the same dict shape aggregate_condition()
    produces, so plot_comparison() doesn't need to know or care whether
    its inputs came fresh from aggregate_condition() or from disk."""
    with open(stats_path) as f:
        stats = json.load(f)
    npz_path = stats_path.parent / 'positions.npz'
    if npz_path.exists():
        npz = np.load(npz_path)
        stats['raw_positions'] = {
            finger: {
                'trial_mean_rel': npz[f'{finger}__trial_mean_rel'],
                'frame_centroid_rel': npz[f'{finger}__frame_centroid_rel'],
                'frame_centroid_variability': npz[f'{finger}__frame_centroid_variability'],
            }
            for finger in FINGERS
        }
    else:
        stats['raw_positions'] = None
    return stats


# ─── Shared visual building blocks ─────────────────────────────────────────────
def _auto_lim(*position_dicts, pad_frac: float = 1.2, min_lim: float = 10.0) -> float:
    """A single symmetric X/Y limit covering every point across however many
    position dicts are passed — so a single-condition plot and a two-
    condition comparison plot never end up on different, hard-to-compare
    scales just because one happened to have a slightly larger max."""
    vals = []
    for pdict in position_dicts:
        for finger in FINGERS:
            arr = np.asarray(pdict.get(finger, []))
            if len(arr):
                vals.append(np.abs(arr[:, :2]).max())
    return max(min_lim, max(vals) * pad_frac) if vals else min_lim


def _banner(ax, text: str, color: str):
    """A full-width colored header bar with bold white centered text —
    same visual role as a slide section header, so each condition's part
    of the figure is unmistakable at a glance."""
    ax.set_facecolor(color)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=15, fontweight='bold',
            color='white', transform=ax.transAxes)


def _scatter_panel(ax, raw_positions: dict, lim: float, title: str):
    for finger in FINGERS:
        pos = np.asarray(raw_positions.get(finger, []))
        if len(pos):
            ax.scatter(pos[:, 0], pos[:, 1], s=10, alpha=0.35, color=FINGER_COLORS[finger],
                       label=finger, edgecolors='none')
    ax.axhline(0, color='#999', linestyle='--', linewidth=1)
    ax.axvline(0, color='#999', linestyle='--', linewidth=1)
    ax.scatter([0], [0], marker='x', color='black', s=100, zorder=5, linewidths=2.2)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11.5)
    ax.grid(alpha=0.2)


def _legend_key_box(ax):
    ax.axis('off')
    ax.text(0.5, 0.97, 'Fingertip\nLandmarks', ha='center', va='top', fontsize=12,
            fontweight='bold', transform=ax.transAxes)
    y = 0.74
    for finger in FINGERS:
        ax.scatter([0.18], [y], s=100, color=FINGER_COLORS[finger], transform=ax.transAxes, clip_on=False)
        ax.text(0.34, y, finger, va='center', fontsize=11, transform=ax.transAxes)
        y -= 0.135
    ax.scatter([0.18], [y], marker='x', s=100, color='black', transform=ax.transAxes,
               clip_on=False, linewidths=2.2)
    ax.text(0.34, y, 'hand centroid\n(origin)', va='center', fontsize=9.5, transform=ax.transAxes)


def _stat_box(ax, title: str, color: str, rows: list):
    """rows: list of (label, value_str). Label and value are stacked
    vertically (not side by side) — side by side was overlapping whenever
    either string was wider than expected for the box, since the two
    ended up sharing the same horizontal band; stacking makes that
    impossible regardless of string length or font metrics."""
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                                facecolor='white', edgecolor=color, linewidth=2, zorder=0))
    ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=11.5, fontweight='bold',
            color=color, transform=ax.transAxes)
    y = 0.80
    step = 0.78 / max(len(rows), 1)
    for label, value in rows:
        ax.text(0.5, y, label, ha='center', va='center', fontsize=10.5, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, y - step * 0.42, value, ha='center', va='center', fontsize=9.5, color='#333',
                transform=ax.transAxes)
        y -= step


def _caption_box(ax, lines: list, title: str = 'How to read this figure'):
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0.01, 0.02), 0.98, 0.96, transform=ax.transAxes,
                                facecolor='#f7f7f7', edgecolor='#ccc', zorder=0))
    ax.text(0.035, 0.88, title, ha='left', va='top', fontsize=11.5, fontweight='bold', transform=ax.transAxes)
    y = 0.68
    for line in lines:
        ax.text(0.04, y, f'\u2022 {line}', ha='left', va='top', fontsize=9.3, transform=ax.transAxes)
        y -= 0.155


def _grouped_bar(ax, values_a, values_b, label_a, label_b, ylabel, title, color_a='#1f77b4', color_b='#2ca02c',
                 annotate_ratio: bool = True):
    x = np.arange(len(FINGERS))
    width = 0.35
    ax.bar(x - width / 2, values_a, width, label=label_a, color=color_a)
    ax.bar(x + width / 2, values_b, width, label=label_b, color=color_b)
    ax.set_xticks(x); ax.set_xticklabels(FINGERS, rotation=20, ha='right')
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11.5, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.25)

    peak = max(max(values_a, default=0), max(values_b, default=0))
    ax.set_ylim(0, peak * 1.22 if peak > 0 else 1.0)
    if annotate_ratio:
        for i, (va, vb) in enumerate(zip(values_a, values_b)):
            if va <= 0:
                continue
            ratio = vb / va
            top = max(va, vb)
            ax.text(x[i], top + peak * 0.04, f'{ratio:.1f}\u00d7', ha='center', va='bottom',
                    fontsize=8.5, color='#555')


def _single_bar(ax, values, ylabel, title):
    ax.bar(FINGERS, values, color=[FINGER_COLORS[f] for f in FINGERS])
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11)
    ax.tick_params(axis='x', rotation=20)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.grid(axis='y', alpha=0.25)


def _pairwise_panel(ax, pair_stats: dict, title: str, color='#888'):
    if not pair_stats:
        ax.axis('off')
        ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes, color='#999')
        return
    pairs = sorted(pair_stats.items(), key=lambda kv: kv[1])
    ax.barh([p for p, _ in pairs], [v for _, v in pairs], color=color)
    ax.set_xlabel('distance std (mm)')
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis='y', labelsize=8.5)
    ax.grid(axis='x', alpha=0.25)


# ─── Plotting: single condition ────────────────────────────────────────────────
def plot_single_condition(agg: dict, label: str, save_path: Path):
    frame_rel = {f: agg['raw_positions'][f]['frame_centroid_rel'] for f in FINGERS}
    lim = _auto_lim(frame_rel)

    fig = plt.figure(figsize=(17, 16), constrained_layout=True)
    gs = fig.add_gridspec(5, 4, height_ratios=[0.35, 3.2, 0.4, 2.3, 1.1])
    color = '#2ca02c'

    _banner(fig.add_subplot(gs[0, :]), f'{label.upper()} FINGER WRITING  (n={agg["n_trials"]} trials)', color)

    ax_scatter = fig.add_subplot(gs[1, 0:2])
    _scatter_panel(ax_scatter, frame_rel, lim, 'Fingertip position relative to per-frame hand centroid')
    ax_scatter.legend(fontsize=8, loc='upper right')

    _legend_key_box(fig.add_subplot(gs[1, 2]))

    stat_rows = [(f, f'{agg["stats"][f]["pos_spread_frame_centroid_rms_mm"]:.1f} mm') for f in FINGERS]
    _stat_box(fig.add_subplot(gs[1, 3]), 'Spread (RMS)\nper finger', color, stat_rows)

    ax_note = fig.add_subplot(gs[2, :]); ax_note.axis('off')
    ax_note.text(0.5, 0.5, 'Larger spread / RMS values indicate greater relative finger motion (articulation) during writing',
                 ha='center', va='center', fontsize=11, style='italic', color='#555', transform=ax_note.transAxes)

    _single_bar(fig.add_subplot(gs[3, 0]), [agg['stats'][f]['accel_rms_mm_s2'] for f in FINGERS],
                'RMS (mm/s\u00b2)', 'Acceleration magnitude')
    _single_bar(fig.add_subplot(gs[3, 1]), [agg['stats'][f]['gyro_rms_rad_s'] for f in FINGERS],
                'RMS (rad/s)', 'Gyro magnitude')
    _single_bar(fig.add_subplot(gs[3, 2]), [agg['stats'][f]['path_length_mm_per_s'] for f in FINGERS],
                'mm/s', 'Tip path length per second')
    _pairwise_panel(fig.add_subplot(gs[3, 3]), agg['pairwise_distance_std_mm'],
                     'Inter-finger distance\nvariability (std, mm)')

    _caption_box(fig.add_subplot(gs[4, :]), [
        'Each point in the scatter is one video frame during writing; the black \u00d7 is that frame\u2019s '
        '5-fingertip centroid (origin) — there is no wrist landmark in the saved data, so this is the '
        'closest available stand-in for "how much the fingers moved relative to the hand."',
        'Position spread, acceleration, gyro, and path length are all independent ways of asking "how '
        'much did this fingertip move" \u2014 shown together since no single one tells the whole story.',
        'Inter-finger distance variability: how much the distance between two fingertips changes over a '
        'trial. Near zero means the whole hand moved as one rigid unit; larger means the fingers moved '
        'independently of each other.',
    ])

    fig.suptitle(f'Fingertip Motion Analysis \u2014 {label}', fontsize=18, fontweight='bold')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def print_table(agg: dict, label: str):
    print(f'\n=== {label} (n={agg["n_trials"]} trials) ===')
    print(f'{"finger":8s} {"spread(frame)":>14s} {"spread(trial)":>14s} {"accel_rms":>11s} {"gyro_rms":>10s} {"path/s":>9s}')
    for finger in FINGERS:
        s = agg['stats'][finger]
        print(f'{finger:8s} {s["pos_spread_frame_centroid_rms_mm"]:14.2f} {s["pos_spread_trial_mean_rms_mm"]:14.2f} '
              f'{s["accel_rms_mm_s2"]:11.1f} {s["gyro_rms_rad_s"]:10.3f} {s["path_length_mm_per_s"]:9.1f}')
    if agg['pairwise_distance_std_mm']:
        print('\npairwise distance std (mm) — larger = fingers move more independently of each other:')
        for pair, val in sorted(agg['pairwise_distance_std_mm'].items(), key=lambda kv: -kv[1]):
            print(f'  {pair:18s} {val:.2f}')


# ─── Plotting: rigid vs. flexible comparison ──────────────────────────────────
def plot_comparison(stats_a: dict, stats_b: dict, save_path: Path):
    label_a, label_b = stats_a['label'], stats_b['label']
    color_a, color_b = '#1f77b4', '#2ca02c'

    pos_a = stats_a.get('raw_positions'); pos_b = stats_b.get('raw_positions')
    have_positions = pos_a is not None and pos_b is not None

    metrics = [
        ('pos_spread_frame_centroid_rms_mm', 'RMS (mm)', 'Spread (frame-centroid-relative)'),
        ('pos_spread_trial_mean_rms_mm', 'RMS (mm)', 'Spread (trial-mean-relative)'),
        ('accel_rms_mm_s2', 'RMS (mm/s\u00b2)', 'Acceleration magnitude'),
        ('gyro_rms_rad_s', 'RMS (rad/s)', 'Gyro magnitude'),
        ('path_length_mm_per_s', 'mm/s', 'Tip path length per second'),
    ]
    n_bar_panels = len(metrics) + 1   # +1 for inter-finger distance variability
    bar_cols = 3
    n_bar_rows = -(-n_bar_panels // bar_cols)   # ceil division

    # Every row this figure needs, decided up front (rather than reusing a
    # row index for two different purposes) — the bug this replaced was
    # exactly that: the caption box and the last row of bar charts were
    # both placed at the same grid row and silently overlapped.
    n_cols = 6 if have_positions else 3
    row_heights = ([0.35, 2.0] if have_positions else []) + [0.55] + [2.3] * n_bar_rows + [1.2]
    n_rows = len(row_heights)

    fig_height = (2.6 if have_positions else 0.0) + 0.8 + 2.5 * n_bar_rows + 1.7
    fig = plt.figure(figsize=(19 if have_positions else 15, fig_height), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=row_heights)

    r = 0
    if have_positions:
        _banner(fig.add_subplot(gs[r, 0:n_cols // 2]), f'{label_a.upper()} FINGER WRITING', color_a)
        _banner(fig.add_subplot(gs[r, n_cols // 2:n_cols]), f'{label_b.upper()} FINGER WRITING', color_b)
        r += 1

        # One zoomed, mean-centered panel per finger — NOT the raw
        # frame-centroid position (that still has each finger's anatomical
        # offset baked in, e.g. thumb sitting ~60mm from the hand centroid
        # regardless of how much it actually moved — at a shared axis scale
        # covering that whole range, a genuine few-mm difference in actual
        # movement between conditions is invisible). Centering each finger
        # on its own mean removes that offset and zooms to a scale sized to
        # that finger specifically, so a real difference in motion shows up
        # as a visibly different cloud size / circle radius, and a genuine
        # non-difference (e.g. thumb position spread was nearly identical
        # between conditions in this dataset) shows up as two similarly-
        # sized clouds — instead of everything looking the same purely
        # because of the shared wide axis scale.
        for i, finger in enumerate(FINGERS):
            ax = fig.add_subplot(gs[r, i])
            va = np.asarray(pos_a[finger]['frame_centroid_variability'])
            vb = np.asarray(pos_b[finger]['frame_centroid_variability'])
            lim = _auto_lim({finger: va}, {finger: vb}, pad_frac=1.35, min_lim=2.0)
            ax.scatter(va[:, 0], va[:, 1], s=7, alpha=0.3, color=color_a, label=label_a, edgecolors='none')
            ax.scatter(vb[:, 0], vb[:, 1], s=7, alpha=0.3, color=color_b, label=label_b, edgecolors='none')
            ra = stats_a['stats'][finger]['pos_spread_frame_centroid_rms_mm']
            rb = stats_b['stats'][finger]['pos_spread_frame_centroid_rms_mm']
            ax.add_patch(plt.Circle((0, 0), ra, fill=False, edgecolor=color_a, linewidth=2))
            ax.add_patch(plt.Circle((0, 0), rb, fill=False, edgecolor=color_b, linewidth=2, linestyle='--'))
            ax.axhline(0, color='#ddd', linewidth=0.8, zorder=0)
            ax.axvline(0, color='#ddd', linewidth=0.8, zorder=0)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            bigger_ratio = max(ra, rb) / min(ra, rb) if min(ra, rb) > 0 else float('nan')
            ax.set_title(f'{finger}\n{ra:.1f} vs {rb:.1f} mm ({bigger_ratio:.1f}\u00d7)', fontsize=9.5)

        ax_legend = fig.add_subplot(gs[r, 5]); ax_legend.axis('off')
        ax_legend.text(0.5, 0.95, 'Motion spread\n(mean-removed)', ha='center', va='top',
                        fontsize=10.5, fontweight='bold', transform=ax_legend.transAxes)
        ax_legend.scatter([0.15], [0.68], s=60, color=color_a, transform=ax_legend.transAxes, clip_on=False)
        ax_legend.text(0.28, 0.68, label_a, va='center', fontsize=9.5, transform=ax_legend.transAxes)
        ax_legend.scatter([0.15], [0.55], s=60, color=color_b, transform=ax_legend.transAxes, clip_on=False)
        ax_legend.text(0.28, 0.55, label_b, va='center', fontsize=9.5, transform=ax_legend.transAxes)
        ax_legend.plot([0.06, 0.24], [0.40, 0.40], color=color_a, linewidth=2, transform=ax_legend.transAxes)
        ax_legend.text(0.28, 0.40, f'{label_a} RMS radius', va='center', fontsize=8.5, transform=ax_legend.transAxes)
        ax_legend.plot([0.06, 0.24], [0.30, 0.30], color=color_b, linewidth=2, linestyle='--',
                       transform=ax_legend.transAxes)
        ax_legend.text(0.28, 0.30, f'{label_b} RMS radius', va='center', fontsize=8.5, transform=ax_legend.transAxes)
        ax_legend.text(0.03, 0.14, 'Circle = that condition\'s spread RMS.\nEach panel is centered and scaled\n'
                                    'to its own finger.', va='top', fontsize=8, color='#666',
                       transform=ax_legend.transAxes)
        r += 1
    else:
        ax_note0 = fig.add_subplot(gs[r, :]); ax_note0.axis('off')
        ax_note0.text(0.5, 0.5, f'{label_a} vs {label_b} \u2014 fingertip motion comparison '
                                 f'(no saved per-point positions available for a scatter plot)',
                      ha='center', va='center', fontsize=13, fontweight='bold', transform=ax_note0.transAxes)
        r += 1

    # Headline: one composite ratio across *all six* metrics (geometric mean
    # — robust to any one metric being an outlier), not just spread alone,
    # plus how many of the six metrics agree on which condition moved more.
    # This is the number meant to answer "so which one actually had more
    # articulation" in one glance, instead of making the reader eyeball six
    # separate bar charts and average it themselves.
    per_metric_ratio = []
    for key, _, _ in metrics:
        va = np.mean([stats_a['stats'][f][key] for f in FINGERS])
        vb = np.mean([stats_b['stats'][f][key] for f in FINGERS])
        if va > 0:
            per_metric_ratio.append(vb / va)
    mean_pw_a = np.mean(list(stats_a['pairwise_distance_std_mm'].values())) if stats_a['pairwise_distance_std_mm'] else None
    mean_pw_b = np.mean(list(stats_b['pairwise_distance_std_mm'].values())) if stats_b['pairwise_distance_std_mm'] else None
    if mean_pw_a:
        per_metric_ratio.append(mean_pw_b / mean_pw_a)

    n_favor_b = sum(1 for x in per_metric_ratio if x > 1)
    n_total = len(per_metric_ratio)
    composite = float(np.exp(np.mean(np.log(per_metric_ratio)))) if per_metric_ratio else float('nan')
    if composite >= 1:
        bigger, smaller, display_ratio, n_favor_bigger = label_b, label_a, composite, n_favor_b
    else:
        bigger, smaller, display_ratio, n_favor_bigger = label_a, label_b, 1.0 / composite, n_total - n_favor_b

    ax_note = fig.add_subplot(gs[r, :]); ax_note.axis('off')
    ax_note.add_patch(plt.Rectangle((0.05, 0.08), 0.9, 0.84, transform=ax_note.transAxes,
                                     facecolor='#fffbe6', edgecolor='#e0c200', linewidth=1.5, zorder=0))
    ax_note.text(0.5, 0.5,
                 f'Overall: "{bigger}" shows {display_ratio:.1f}\u00d7 more fingertip motion than "{smaller}" '
                 f'on average \u2014 {n_favor_bigger}/{n_total} metrics agree',
                 ha='center', va='center', fontsize=13, fontweight='bold', color='#7a5c00',
                 transform=ax_note.transAxes)
    r += 1

    bar_row_start = r
    col_width = n_cols // bar_cols   # 2 if have_positions else 1
    for i, (key, ylabel, title) in enumerate(metrics):
        row = bar_row_start + i // bar_cols
        col0 = (i % bar_cols) * col_width
        ax = fig.add_subplot(gs[row, col0:col0 + col_width])
        vals_a = [stats_a['stats'][f][key] for f in FINGERS]
        vals_b = [stats_b['stats'][f][key] for f in FINGERS]
        _grouped_bar(ax, vals_a, vals_b, label_a, label_b, ylabel, title, color_a, color_b)

    # 6th panel: inter-finger distance variability, averaged over all pairs
    i = len(metrics)
    row = bar_row_start + i // bar_cols
    col0 = (i % bar_cols) * col_width
    ax_pw = fig.add_subplot(gs[row, col0:col0 + col_width])
    mean_a = mean_pw_a or 0.0
    mean_b = mean_pw_b or 0.0
    ax_pw.bar([label_a, label_b], [mean_a, mean_b], color=[color_a, color_b])
    ax_pw.set_ylabel('mean pairwise distance std (mm)')
    ax_pw.set_title('Inter-finger distance variability\n(avg over all finger pairs)', fontsize=11.5, fontweight='bold')
    ax_pw.grid(axis='y', alpha=0.25)
    if mean_a > 0:
        peak = max(mean_a, mean_b)
        ax_pw.set_ylim(0, peak * 1.22)
        ax_pw.text(0.5, peak * 1.04, f'{mean_b / mean_a:.1f}\u00d7', ha='center', fontsize=9, color='#555')
    r = bar_row_start + n_bar_rows

    _caption_box(fig.add_subplot(gs[r, :]), [
        'Every panel compares the same trials-aggregated statistic between the two conditions \u2014 '
        f'blue = {label_a}, green = {label_b} throughout.',
        'Spread and inter-finger distance variability use a per-frame 5-fingertip centroid as the '
        'reference point (no wrist landmark is saved in the data) \u2014 they isolate relative finger '
        'motion from whatever the whole hand/wrist is doing.',
        'Acceleration, gyro, and path length are direct sensor/motion magnitudes, independent of any '
        'reference point choice.',
    ], title='How to read this comparison')

    fig.suptitle(f'{label_a} vs {label_b} \u2014 Fingertip Motion Comparison', fontsize=19, fontweight='bold')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=170, bbox_inches='tight')
    plt.close(fig)

    print(f'\n{"metric":34s} {label_a:>14s} {label_b:>14s} {"ratio(b/a)":>11s}')
    for key, _, title in metrics:
        va = np.mean([stats_a['stats'][f][key] for f in FINGERS])
        vb = np.mean([stats_b['stats'][f][key] for f in FINGERS])
        ratio = vb / va if va else float('nan')
        print(f'{title:34s} {va:14.2f} {vb:14.2f} {ratio:11.2f}')
    print(f'{"Inter-finger dist. variability":34s} {mean_a:14.2f} {mean_b:14.2f} '
          f'{(mean_b / mean_a if mean_a else float("nan")):11.2f}')


def main():
    parser = argparse.ArgumentParser(description='Analyze fingertip motion/articulation from fingertip_imu.csv')
    parser.add_argument('--dataset-root', type=Path, default=None)
    parser.add_argument('--classes', nargs='+', default=['digits_0', 'digits_1', 'digits_2'])
    parser.add_argument('--label', default='condition')
    parser.add_argument('--out-dir', type=Path, default=None,
                         help='where to save stats.json, positions.npz, and this condition\'s own figure')
    parser.add_argument('--conditions', nargs='+', default=None,
                         help='for a --dataset-root containing one subfolder per condition (e.g. '
                              'dataset/flexible/, dataset/rigid/, each with digits_0/1/2 inside): '
                              'analyzes every condition given and, if exactly two are given, also '
                              'produces the full comparison — all in one command, e.g. '
                              '--conditions flexible rigid')
    parser.add_argument('--compare', nargs=2, type=Path, default=None, metavar=('STATS_A', 'STATS_B'),
                         help='two stats.json files (from earlier --out-dir runs) to compare directly, '
                              'skipping re-analysis')
    parser.add_argument('--compare-out', type=Path, default=Path('rigid_vs_flexible.png'))
    args = parser.parse_args()

    if args.compare is not None:
        stats_a = load_condition(args.compare[0])
        stats_b = load_condition(args.compare[1])
        plot_comparison(stats_a, stats_b, args.compare_out)
        print(f'\n[PLOT] comparison saved to {args.compare_out}')
        return

    if args.conditions is not None:
        if args.dataset_root is None or args.out_dir is None:
            parser.error('--conditions requires both --dataset-root and --out-dir')
        loaded = {}
        for cond in args.conditions:
            cond_root = args.dataset_root / cond
            print(f'\n[CONDITION] "{cond}"  ({cond_root})')
            agg = aggregate_condition(cond_root, args.classes)
            print_table(agg, cond)
            cond_out = args.out_dir / cond
            save_condition(agg, cond_out, cond)
            plot_single_condition(agg, cond, cond_out / f'{cond}_summary.png')
            print(f'[SAVED] {cond_out}/stats.json, {cond_out}/positions.npz, {cond_out}/{cond}_summary.png')
            loaded[cond] = load_condition(cond_out / 'stats.json')

        if len(args.conditions) == 2:
            compare_out = args.out_dir / f'{args.conditions[0]}_vs_{args.conditions[1]}.png'
            plot_comparison(loaded[args.conditions[0]], loaded[args.conditions[1]], compare_out)
            print(f'\n[PLOT] comparison saved to {compare_out}')
        elif len(args.conditions) > 2:
            print(f'\n[NOTE] {len(args.conditions)} conditions analyzed individually — '
                  f'plot_comparison() only handles two at a time, so pick a pair and use --compare '
                  f'for any side-by-side comparison beyond the first two.')
        return

    if args.dataset_root is None:
        parser.error('--dataset-root is required unless using --compare or --conditions')
    if args.out_dir is None:
        parser.error('--out-dir is required unless using --compare or --conditions')

    agg = aggregate_condition(args.dataset_root, args.classes)
    print_table(agg, args.label)
    save_condition(agg, args.out_dir, args.label)
    fig_path = args.out_dir / f'{args.label}_summary.png'
    plot_single_condition(agg, args.label, fig_path)
    print(f'\n[SAVED] {args.out_dir}/stats.json, {args.out_dir}/positions.npz, {fig_path}')


if __name__ == '__main__':
    main()