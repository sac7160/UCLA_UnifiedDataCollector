"""
browse_trajectories.py

Paginated, multi-page PDF contact sheet of fingertip trajectories, split into
per-touch strokes and rendered with a light->dark color gradient plus
repeated direction arrows along each stroke, so the FULL path of motion is
visible, not just start/end points.

Stroke boundaries, per trial, are resolved in this priority order:
    1. manual_range.json   (from select_trim_range.py, if you made one)
    2. events.csv           (audio_touch_on / audio_touch_off)
    3. whole trajectory as a single stroke (fallback)

Prefers trajectory_smooth.csv (from extract_fingertip_trajectory.py) over the
noisier live-tracked trajectory.csv, if present in the trial folder.

Usage:
    python browse_trajectories.py --dataset-root dataset --label d --out browse_d.pdf
    python browse_trajectories.py --dataset-root dataset --label d --sample 60 --out browse_d.pdf
"""
import argparse
import glob
import json
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

PREFERRED_TRAJ_FILENAMES = ["trajectory_smooth_120hz.csv", "trajectory.csv"]
PREFERRED_COORD_COLS = [
    ("local_x_mm", "local_y_mm"),
    ("pos_x_mm", "pos_y_mm"),
    ("x_px", "y_px"),
]
STROKE_CMAPS = ["Greens", "Oranges", "Blues", "RdPu", "Purples", "YlOrBr"]


def find_trial_dirs(dataset_root, label, participants=None):
    patterns = [
        os.path.join(dataset_root, "*", "dataset", label, "trial_*"),
        os.path.join(dataset_root, label, "trial_*"),
    ]
    dirs = []
    for pat in patterns:
        dirs.extend(glob.glob(pat))
    dirs = sorted(set(d for d in dirs if os.path.isdir(d)))
    if participants:
        dirs = [d for d in dirs if any(f"{os.sep}{p}{os.sep}" in d for p in participants)]
    return dirs


def resolve_traj_path(trial_dir):
    for name in PREFERRED_TRAJ_FILENAMES:
        p = os.path.join(trial_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No trajectory csv in {trial_dir}")


def parse_ids(trial_dir):
    parts = trial_dir.split(os.sep)
    trial = parts[-1]
    participant = "?"
    for i, p in enumerate(parts):
        if p == "dataset" and i > 0:
            participant = parts[i - 1]
            break
    return participant, trial


def load_trajectory(csv_path):
    df = pd.read_csv(csv_path)
    if "detected" in df.columns:
        df = df[df["detected"] == 1]
    x_col = y_col = None
    for xc, yc in PREFERRED_COORD_COLS:
        if xc in df.columns and yc in df.columns:
            x_col, y_col = xc, yc
            break
    if x_col is None:
        raise ValueError(f"No known coordinate columns in {csv_path}: {list(df.columns)}")
    t_col = "time_aligned" if "time_aligned" in df.columns else df.columns[0]
    return df[t_col].to_numpy(float), df[x_col].to_numpy(float), df[y_col].to_numpy(float)


def ranges_from_events(events_path):
    ev = pd.read_csv(events_path)
    pairs, on_time = [], None
    for _, row in ev.sort_values("time_aligned").iterrows():
        if row["event"] == "audio_touch_on":
            on_time = row["time_aligned"]
        elif row["event"] == "audio_touch_off" and on_time is not None:
            pairs.append((on_time, row["time_aligned"]))
            on_time = None
    return pairs


def load_strokes(trial_dir):
    traj_path = resolve_traj_path(trial_dir)
    t, x, y = load_trajectory(traj_path)

    manual_path = os.path.join(trial_dir, "manual_range.json")
    ranges = []
    if os.path.exists(manual_path):
        with open(manual_path) as f:
            ranges = [tuple(r) for r in json.load(f).get("ranges", [])]
    if not ranges:
        events_path = os.path.join(trial_dir, "events.csv")
        if os.path.exists(events_path):
            ranges = ranges_from_events(events_path)

    if not ranges:
        return [(x, y)]

    strokes = []
    for on_t, off_t in ranges:
        mask = (t >= on_t) & (t <= off_t)
        if mask.sum() >= 2:
            strokes.append((x[mask], y[mask]))
    return strokes if strokes else [(x, y)]


def plot_stroke_gradient(ax, x, y, cmap_name, arrow_every=10, linewidth=2.4):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    progress = np.linspace(0.15, 1.0, len(x))
    lc = LineCollection(segments, cmap=cmap_name, norm=Normalize(0, 1))
    lc.set_array(progress[:-1])
    lc.set_linewidth(linewidth)
    ax.add_collection(lc)

    cmap = plt.get_cmap(cmap_name)
    for i in range(1, len(x) - 1, max(arrow_every, 1)):
        ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i - 1], y[i - 1]),
                    arrowprops=dict(arrowstyle="-|>", color=cmap(progress[i]),
                                     lw=0.7, mutation_scale=6))


def plot_trial(ax, trial_dir, arrow_every=10):
    strokes = load_strokes(trial_dir)
    all_x = np.concatenate([s[0] for s in strokes])
    all_y = np.concatenate([s[1] for s in strokes])
    for i, (x, y) in enumerate(strokes):
        cmap_name = STROKE_CMAPS[i % len(STROKE_CMAPS)]
        plot_stroke_gradient(ax, x, y, cmap_name, arrow_every=max(arrow_every, len(x) // 4))
    ax.set_xlim(all_x.min() - 5, all_x.max() + 5)
    ax.set_ylim(all_y.max() + 5, all_y.min() - 5)  # inverted (image coords)
    ax.set_aspect("equal")
    ax.axis("off")
    return len(strokes)


def main():
    ap = argparse.ArgumentParser(description="Stroke-aware paginated contact-sheet browser.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--participants", nargs="*", default=None)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--per-page", type=int, default=20)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="browse.pdf")
    args = ap.parse_args()

    trial_dirs = find_trial_dirs(args.dataset_root, args.label, args.participants)
    if not trial_dirs:
        print(f"[WARN] No trial folders found for label='{args.label}'.")
        return

    if args.sample and len(trial_dirs) > args.sample:
        random.seed(args.seed)
        trial_dirs = sorted(random.sample(trial_dirs, args.sample))
        print(f"[INFO] Sampled {args.sample} of the full set for browsing.")

    n = len(trial_dirs)
    cols = args.cols
    per_page = args.per_page
    rows_per_page = math.ceil(per_page / cols)
    n_pages = math.ceil(n / per_page)

    out_path = args.out if args.out.endswith(".pdf") else args.out + ".pdf"
    with PdfPages(out_path) as pdf:
        for page in range(n_pages):
            chunk = trial_dirs[page * per_page:(page + 1) * per_page]
            fig, axes = plt.subplots(rows_per_page, cols, figsize=(cols * 2.2, rows_per_page * 2.2))
            axes = axes.flatten() if per_page > 1 else [axes]

            for ax, trial_dir in zip(axes, chunk):
                try:
                    n_strokes = plot_trial(ax, trial_dir)
                except Exception as e:
                    ax.set_title(f"error: {e}", fontsize=5, color="red", wrap=True)
                    ax.axis("off")
                    continue
                participant, trial = parse_ids(trial_dir)
                has_manual = os.path.exists(os.path.join(trial_dir, "manual_range.json"))
                tag = " [manual]" if has_manual else ""
                ax.set_title(f"{participant}/{trial} ({n_strokes} strokes){tag}", fontsize=6)

            for ax in axes[len(chunk):]:
                ax.axis("off")

            fig.suptitle(
                f"label='{args.label}'  page {page + 1}/{n_pages}  (showing {len(chunk)} of {n})  "
                f"\u2014 light\u2192dark = time progress, color family = stroke order",
                fontsize=8,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[DONE] Saved {n_pages} page(s) covering {n} trials to {out_path}")


if __name__ == "__main__":
    main()