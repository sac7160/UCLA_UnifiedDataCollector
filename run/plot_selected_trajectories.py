"""
plot_selected_trajectories.py

Render the final, publication-quality stroke-variation figure. Each trial is
split into strokes and drawn as a light->dark color gradient (time progress)
with repeated direction arrows, so both stroke ORDER (color family) and the
full DIRECTION of motion within each stroke are visible.

Stroke boundaries, per trial, are resolved in this priority order:
    1. manual_range.json   (from select_trim_range.py)
    2. events.csv           (audio_touch_on / audio_touch_off)
    3. whole trajectory as a single stroke (fallback)

Prefers trajectory_smooth.csv (see extract_fingertip_trajectory.py) over the
noisier live-tracked trajectory.csv, if present.

Edit SELECTIONS below with the (participant, trial_dir) pairs you picked for
each letter, then run:

    python plot_selected_trajectories.py --dataset-root dataset --out figure_stroke_variation.pdf
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

# ---- EDIT THIS: your curated picks, one row per letter, same #cols per row ----
SELECTIONS = {
    "d": [("p4", "trial_005"), ("p14", "trial_005"), ("p11", "trial_001")],
    "f": [("p14", "trial_004"), ("p3", "trial_001"), ("p9", "trial_003")],
    "o": [("p4", "trial_003"), ("p5", "trial_002"), ("p3", "trial_006")],
}
# --------------------------------------------------------------------------

PREFERRED_TRAJ_FILENAMES = ["trajectory_smooth_120hz.csv", "trajectory.csv"]
PREFERRED_COORD_COLS = [
    ("local_x_mm", "local_y_mm"),
    ("pos_x_mm", "pos_y_mm"),
    ("x_px", "y_px"),
]
STROKE_CMAPS = ["Greens", "Oranges", "Blues", "RdPu", "Purples", "YlOrBr"]
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


def resolve_trial_dir(dataset_root, label, participant, trial):
    candidates = [
        os.path.join(dataset_root, participant, "dataset", label, trial),
        os.path.join(dataset_root, label, trial),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(f"No trial folder found for {participant}/{label}/{trial}")


def resolve_traj_path(trial_dir):
    for name in PREFERRED_TRAJ_FILENAMES:
        p = os.path.join(trial_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No trajectory csv in {trial_dir}")


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


def normalize_strokes(strokes):
    all_x = np.concatenate([s[0] for s in strokes])
    all_y = np.concatenate([s[1] for s in strokes])
    cx, cy = all_x.mean(), all_y.mean()
    extent = max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
    if extent == 0:
        extent = 1.0
    return [((x - cx) / extent, (y - cy) / extent) for x, y in strokes]


def plot_stroke_gradient(ax, x, y, cmap_name, arrow_every, linewidth=3.2):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    progress = np.linspace(0.15, 1.0, len(x))
    lc = LineCollection(segments, cmap=cmap_name, norm=Normalize(0, 1))
    lc.set_array(progress[:-1])
    lc.set_linewidth(linewidth)
    lc.set_capstyle("round")
    ax.add_collection(lc)

    cmap = plt.get_cmap(cmap_name)
    for i in range(1, len(x) - 1, max(arrow_every, 1)):
        ax.annotate("", xy=(x[i + 1], y[i + 1]), xytext=(x[i - 1], y[i - 1]),
                    arrowprops=dict(arrowstyle="-|>", color=cmap(progress[i]),
                                     lw=1.4, mutation_scale=11))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out", default="figure_stroke_variation.pdf")
    ap.add_argument("--also-png", action="store_true")
    args = ap.parse_args()

    set_paper_style()

    letters = list(SELECTIONS.keys())
    ncols = max(len(v) for v in SELECTIONS.values())
    nrows = len(letters)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.9, nrows * 1.9))
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    pad = 0.12
    max_strokes_used = 1
    for row_idx, letter in enumerate(letters):
        picks = SELECTIONS[letter]
        for col_idx in range(ncols):
            ax = axes[row_idx][col_idx]
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(PANEL_EDGE)
                spine.set_linewidth(0.6)
            ax.set_xticks([])
            ax.set_yticks([])

            if col_idx >= len(picks):
                ax.axis("off")
                continue

            participant, trial = picks[col_idx]
            trial_dir = resolve_trial_dir(args.dataset_root, letter, participant, trial)
            strokes = normalize_strokes(load_strokes(trial_dir))
            max_strokes_used = max(max_strokes_used, len(strokes))

            for i, (x, y) in enumerate(strokes):
                cmap_name = STROKE_CMAPS[i % len(STROKE_CMAPS)]
                plot_stroke_gradient(ax, x, y, cmap_name, arrow_every=max(len(x) // 5, 1))
                ax.text(x[0], y[0], str(i + 1), fontsize=7,
                        color=plt.get_cmap(cmap_name)(0.9), ha="right", va="bottom",
                        fontweight="bold")

            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_xlim(-0.5 - pad, 0.5 + pad)
            ax.set_ylim(0.5 + pad, -0.5 - pad)

            if row_idx == 0:
                ax.set_title(f"Participant {chr(65 + col_idx)}", fontsize=10, pad=6)
            if col_idx == 0:
                ax.text(-0.34, 0.5, letter, transform=ax.transAxes,
                        fontsize=15, fontweight="bold", ha="center", va="center",
                        fontstyle="italic")

    # Graphical legend: one small light->dark gradient swatch per stroke
    # order, labeled "stroke k", instead of a plain text-only note.
    n_swatches = max_strokes_used
    swatch_w, gap, total_w = 0.16, 0.03, None
    total_w = n_swatches * swatch_w + (n_swatches - 1) * gap
    start_x = 0.5 - total_w / 2
    gradient = np.linspace(0.15, 1.0, 256).reshape(1, -1)
    for i in range(n_swatches):
        cmap_name = STROKE_CMAPS[i % len(STROKE_CMAPS)]
        left = start_x + i * (swatch_w + gap)
        cax = fig.add_axes([left, 0.045, swatch_w, 0.018])
        cax.imshow(gradient, aspect="auto", cmap=plt.get_cmap(cmap_name))
        cax.set_xticks([])
        cax.set_yticks([])
        for spine in cax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color(PANEL_EDGE)
        fig.text(left + swatch_w / 2, 0.028, f"stroke {i + 1}",
                  ha="center", va="top", fontsize=7.5, color=plt.get_cmap(cmap_name)(0.9))

    fig.text(0.5, 0.005,
              "light \u2192 dark = time progress within a stroke",
              ha="center", fontsize=7.5, color="#444441")

    fig.tight_layout(rect=[0.03, 0.09, 1, 1])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[DONE] Saved figure to {args.out}")

    if args.also_png:
        png_path = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(png_path, dpi=400, bbox_inches="tight")
        print(f"[DONE] Saved high-res PNG to {png_path}")


if __name__ == "__main__":
    main()