"""
plot_trajectory_segment.py

Plot just the segment(s) of a trial's trajectory you want, in paper-quality
matplotlib style -- a single clean panel with a light->dark gradient (time
progress) and periodic direction arrows, plus start/end markers.

Which time range to plot is resolved in this priority order:
    1. --start / --end, if you pass them explicitly (single range)
    2. manual_range.json, if present (from select_trim_range.py; can hold
       multiple ranges -> each drawn as its own colored stroke)
    3. events.csv touch-on/off pairs (same multi-stroke behavior)
    4. the entire trajectory, if none of the above are available

Data source priority: trajectory_smooth_120hz.csv > trajectory_smooth.csv >
trajectory.csv (raw). Pass --source to control units:
    smoothed_mm (default) -- clean smoothed trajectory, rescaled to real mm
                              using a scale derived from this trial's own
                              calibrated trajectory.csv
    pixel                 -- raw pixel coordinates, no mm conversion
    calibrated            -- the original (jittery) trajectory.csv in mm

Usage:
    # Explicit time window
    python plot_trajectory_segment.py --trial-dir dataset/p1/dataset/d/trial_005 \
        --start 1.2 --end 2.8 --out figure_trajectory_segment.pdf --also-png

    # Whatever manual_range.json / events.csv already defines
    python plot_trajectory_segment.py --trial-dir dataset/p1/dataset/d/trial_005 \
        --out figure_trajectory_segment.pdf
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.spatial import ConvexHull, QhullError

PREFERRED_TRAJ_FILENAMES = ["trajectory_smooth_120hz.csv", "trajectory_smooth.csv", "trajectory.csv"]
PX_COORD_COLS = ("x_px", "y_px")
MM_COORD_COLS = ("local_x_mm", "local_y_mm")
STROKE_CMAPS = ["Blues", "Oranges", "Greens", "RdPu", "Purples", "YlOrBr"]
PANEL_EDGE = "#B0AFA8"


def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def resolve_trial_dir(args):
    if args.trial_dir:
        return args.trial_dir
    return os.path.join(args.dataset_root, args.participant, "dataset", args.label, args.trial)


def resolve_traj_path(trial_dir, source):
    if source in ("calibrated", "smoothed_mm"):
        p = os.path.join(trial_dir, "trajectory.csv")
        mm_path = p if os.path.exists(p) else None
    else:
        mm_path = None
    if source == "calibrated":
        return mm_path
    for name in PREFERRED_TRAJ_FILENAMES:
        p = os.path.join(trial_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_full_trajectory(trial_dir, source):
    traj_path = resolve_traj_path(trial_dir, source)
    if traj_path is None:
        raise FileNotFoundError(f"No usable trajectory file in {trial_dir} for source='{source}'")

    df = pd.read_csv(traj_path)
    if "detected" in df.columns:
        df = df[df["detected"] == 1]

    x_col, y_col = MM_COORD_COLS if source == "calibrated" else PX_COORD_COLS
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found in {traj_path}")

    t = df["time_aligned"].to_numpy(float) if "time_aligned" in df.columns else np.arange(len(df))
    x = df[x_col].to_numpy(float)
    y = df[y_col].to_numpy(float)

    if source == "smoothed_mm":
        mm_path = os.path.join(trial_dir, "trajectory.csv")
        if os.path.exists(mm_path):
            mm_df = pd.read_csv(mm_path)
            if "detected" in mm_df.columns:
                mm_df = mm_df[mm_df["detected"] == 1]
            if MM_COORD_COLS[0] in mm_df.columns:
                area_px = hull_area(x, y)
                area_mm = hull_area(mm_df[MM_COORD_COLS[0]].to_numpy(float),
                                      mm_df[MM_COORD_COLS[1]].to_numpy(float))
                if area_px > 0:
                    scale = np.sqrt(area_mm / area_px)
                    x, y = x * scale, y * scale
        else:
            print(f"[WARN] No trajectory.csv found for mm-scale derivation in {trial_dir}; "
                  f"falling back to raw pixel units.")

    return t, x, y


def hull_area(x, y):
    try:
        return ConvexHull(np.column_stack([x, y])).volume
    except QhullError:
        return (x.max() - x.min()) * (y.max() - y.min())


def resolve_ranges(args, trial_dir):
    if args.start is not None and args.end is not None:
        return [(args.start, args.end)]

    manual_path = os.path.join(trial_dir, "manual_range.json")
    if os.path.exists(manual_path):
        with open(manual_path) as f:
            ranges = [tuple(r) for r in json.load(f).get("ranges", [])]
        if ranges:
            return ranges

    events_path = os.path.join(trial_dir, "events.csv")
    if os.path.exists(events_path):
        ev = pd.read_csv(events_path)
        pairs, on_time = [], None
        for _, row in ev.sort_values("time_aligned").iterrows():
            if row["event"] == "audio_touch_on":
                on_time = row["time_aligned"]
            elif row["event"] == "audio_touch_off" and on_time is not None:
                pairs.append((on_time, row["time_aligned"]))
                on_time = None
        if pairs:
            return pairs

    return None  # signals "use everything"


def plot_stroke_gradient(ax, x, y, cmap_name, linewidth=5.5):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    progress = np.linspace(0.15, 1.0, len(x))
    lc = LineCollection(segments, cmap=cmap_name, norm=Normalize(0, 1))
    lc.set_array(progress[:-1])
    lc.set_linewidth(linewidth)
    lc.set_capstyle("round")
    ax.add_collection(lc)

    ax.plot(x[0], y[0], "o", color="#2E7D32", markersize=7, zorder=5,
             markeredgecolor="white", markeredgewidth=0.7)
    ax.plot(x[-1], y[-1], "o", color="#C62828", markersize=7, zorder=5,
             markeredgecolor="white", markeredgewidth=0.7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-dir", default=None)
    ap.add_argument("--dataset-root")
    ap.add_argument("--participant")
    ap.add_argument("--label")
    ap.add_argument("--trial")
    ap.add_argument("--start", type=float, default=None, help="Start time (s) of the segment to plot")
    ap.add_argument("--end", type=float, default=None, help="End time (s) of the segment to plot")
    ap.add_argument("--source", choices=["pixel", "calibrated", "smoothed_mm"], default="smoothed_mm")
    ap.add_argument("--out", default="figure_trajectory_segment.pdf")
    ap.add_argument("--also-png", action="store_true")
    args = ap.parse_args()

    trial_dir = resolve_trial_dir(args)
    if not os.path.isdir(trial_dir):
        raise SystemExit(f"Trial folder not found: {trial_dir}")

    set_paper_style()
    unit = "px" if args.source == "pixel" else "mm"

    t, x, y = load_full_trajectory(trial_dir, args.source)
    ranges = resolve_ranges(args, trial_dir)
    if ranges is None:
        ranges = [(t.min(), t.max())]

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    all_x, all_y = [], []
    for i, (start_t, end_t) in enumerate(ranges):
        mask = (t >= start_t) & (t <= end_t)
        if mask.sum() < 2:
            print(f"[WARN] Range {start_t:.2f}-{end_t:.2f}s has too few points ({mask.sum()}); skipping.")
            continue
        xs, ys = x[mask], y[mask]
        all_x.append(xs)
        all_y.append(ys)
        cmap_name = STROKE_CMAPS[i % len(STROKE_CMAPS)]
        plot_stroke_gradient(ax, xs, ys, cmap_name)

    if not all_x:
        raise SystemExit("No data points fell inside the requested range(s).")

    all_x, all_y = np.concatenate(all_x), np.concatenate(all_y)
    cx, cy = (all_x.max() + all_x.min()) / 2, (all_y.max() + all_y.min()) / 2
    half_extent = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * 1.12 + 1
    ax.set_xlim(cx - half_extent, cx + half_extent)
    ax.set_ylim(cy + half_extent, cy - half_extent)  # inverted (image/mm y-down convention)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.set_xlabel(f"x ({unit})", fontsize=9)
    ax.set_ylabel(f"y ({unit})", fontsize=9)
    ax.tick_params(labelsize=7.5)
    ax.grid(alpha=0.15, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color(PANEL_EDGE)

    range_str = ", ".join(f"{s:.2f}-{e:.2f}s" for s, e in ranges)
    participant = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(trial_dir))))
    label = os.path.basename(os.path.dirname(trial_dir))
    trial = os.path.basename(trial_dir)
    ax.set_title(f"{participant} / '{label}' / {trial}\n{range_str}", fontsize=9)

    fig.text(0.5, 0.005, "light \u2192 dark = time progress; green/red = stroke start/end",
              ha="center", fontsize=7.5, color="#444441")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[DONE] Saved figure to {args.out}")
    if args.also_png:
        png_path = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(png_path, dpi=400, bbox_inches="tight")
        print(f"[DONE] Saved high-res PNG to {png_path}")


if __name__ == "__main__":
    main()