"""
plot_size_variation.py

Compare how large each participant's handwriting is, using ALL of their
trials for a given label (not a hand-picked few). For each trial, the
writing region's area is estimated with a convex hull; areas are then
averaged per participant. Two views are produced side by side:

    (a) each participant's actual convex-hull SHAPE overlaid at true scale
        (not a generic circle) -- the trial whose area is closest to that
        participant's mean is used as the representative shape, so both
        size AND aspect ratio / skew (e.g. wide-and-short vs. tall-and-
        narrow writing) stay visible. Pass --show-all-trials to instead
        draw every trial's hull outline (unfilled) per participant, to see
        within-participant spread as well.
    (b) a bar chart of mean +/- std area per participant, for the exact
        numbers to cite in text

Data source priority (see PREFERRED_TRAJ_FILENAMES): trajectory_smooth_120hz.csv
(re-tracked at 120fps) is used if present, falling back to trajectory_smooth.csv,
then the original trajectory.csv.

NOTE on units: the re-tracked trajectory files are in PIXELS, not real-world
mm. This is fine for a *relative* size comparison as long as the camera
framing was identical across all sessions. If you're not confident of that
(e.g. the camera was ever unplugged/remounted between sessions), pass
--source calibrated to use the local_x_mm/local_y_mm columns from the
original trajectory.csv instead, which are in real-world mm regardless of
camera framing.

Usage:
    # Pixel-space area from the 120Hz re-tracked trajectories (default)
    python plot_size_variation.py --dataset-root dataset --label d --out figure_size_variation.pdf

    # Real-world mm^2 area instead (safer if camera framing varied across sessions)
    python plot_size_variation.py --dataset-root dataset --label d --source calibrated --out figure_size_variation.pdf

    # Restrict to specific participants
    python plot_size_variation.py --dataset-root dataset --label d --participants p1 p2 p3 --out figure_size_variation.pdf
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull, QhullError

PREFERRED_TRAJ_FILENAMES = ["trajectory_smooth_120hz.csv", "trajectory_smooth.csv", "trajectory.csv"]
PX_COORD_COLS = ("x_px", "y_px")
MM_COORD_COLS = ("local_x_mm", "local_y_mm")

PALETTE = ["#1D9E75", "#D85A30", "#378ADD", "#BA7517", "#7F77DD", "#D4537E",
           "#4E9A9A", "#B23A3A", "#6C8E3C", "#8A5FB0"]
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


def find_trial_dirs(dataset_root, label=None, participants=None):
    label_pattern = label if label else "*"
    patterns = [
        os.path.join(dataset_root, "*", "dataset", label_pattern, "trial_*"),
        os.path.join(dataset_root, label_pattern, "trial_*"),
    ]
    dirs = []
    for pat in patterns:
        dirs.extend(glob.glob(pat))
    dirs = sorted(set(d for d in dirs if os.path.isdir(d)))
    if participants:
        dirs = [d for d in dirs if any(f"{os.sep}{p}{os.sep}" in d for p in participants)]
    return dirs


def parse_participant(trial_dir):
    parts = trial_dir.split(os.sep)
    for i, p in enumerate(parts):
        if p == "dataset" and i > 0:
            return parts[i - 1]
    return "?"


def resolve_traj_path(trial_dir, source):
    if source in ("calibrated", "smoothed_mm_ref"):
        p = os.path.join(trial_dir, "trajectory.csv")
        return p if os.path.exists(p) else None
    for name in PREFERRED_TRAJ_FILENAMES:
        p = os.path.join(trial_dir, name)
        if os.path.exists(p):
            return p
    return None


def load_ranges(trial_dir):
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
        return pairs
    return None


def load_writing_xy(trial_dir, source):
    """source: 'pixel' (smoothed 120Hz px), 'calibrated' (raw mm), or
    'smoothed_mm' (smoothed px rescaled to mm using a per-trial scale
    derived from that trial's own calibrated trajectory.csv)."""
    if source == "smoothed_mm":
        return load_writing_xy_smoothed_mm(trial_dir)

    traj_path = resolve_traj_path(trial_dir, source)
    if traj_path is None:
        return None

    df = pd.read_csv(traj_path)
    if "detected" in df.columns:
        df = df[df["detected"] == 1]

    x_col, y_col = MM_COORD_COLS if source == "calibrated" else PX_COORD_COLS
    if x_col not in df.columns:
        return None

    t = df["time_aligned"].to_numpy(float) if "time_aligned" in df.columns else np.arange(len(df))
    x = df[x_col].to_numpy(float)
    y = df[y_col].to_numpy(float)

    ranges = load_ranges(trial_dir)
    if ranges:
        mask = np.zeros(len(t), dtype=bool)
        for on_t, off_t in ranges:
            mask |= (t >= on_t) & (t <= off_t)
        if mask.sum() >= 3:
            return x[mask], y[mask]
    return (x, y) if len(x) >= 3 else None


def load_writing_xy_smoothed_mm(trial_dir):
    """Combine the clean smoothed pixel trajectory with a per-trial px->mm
    scale derived by comparing hull areas against that same trial's
    calibrated (mm) trajectory.csv. This corrects for camera framing that
    may differ across sessions, since the scale is derived per trial rather
    than assumed globally."""
    px_xy = load_writing_xy(trial_dir, "pixel")
    mm_xy = load_writing_xy(trial_dir, "calibrated")
    if px_xy is None or mm_xy is None:
        return None

    area_px, _ = hull_area_and_vertices(*px_xy)
    area_mm, _ = hull_area_and_vertices(*mm_xy)
    if area_px <= 0:
        return None

    scale = np.sqrt(area_mm / area_px)  # mm per px, derived from this trial only
    x_px, y_px = px_xy
    return x_px * scale, y_px * scale


def hull_area_and_vertices(x, y):
    """Convex hull area and boundary vertices (closed polygon, centered on
    the point cloud's own centroid). Falls back to the bounding box if the
    hull is degenerate (near-collinear points, very few points, etc.)."""
    pts = np.column_stack([x, y])
    cx, cy = x.mean(), y.mean()
    try:
        hull = ConvexHull(pts)
        area = hull.volume  # 'volume' is the 2D area for a 2D hull
        verts = pts[hull.vertices]
    except QhullError:
        area = (x.max() - x.min()) * (y.max() - y.min())
        verts = np.array([
            [x.min(), y.min()], [x.max(), y.min()],
            [x.max(), y.max()], [x.min(), y.max()],
        ])
    return area, verts - [cx, cy]


def radial_profile(verts, query_angles):
    """Represent a centered convex polygon's boundary as radius-vs-angle
    (valid since a convex shape is star-shaped around its own centroid),
    then sample it at a fixed set of angles for cross-trial averaging."""
    ang = np.arctan2(verts[:, 1], verts[:, 0])
    r = np.hypot(verts[:, 0], verts[:, 1])
    order = np.argsort(ang)
    ang, r = ang[order], r[order]
    ang_ext = np.concatenate([ang - 2 * np.pi, ang, ang + 2 * np.pi])
    r_ext = np.concatenate([r, r, r])
    return np.interp(query_angles, ang_ext, r_ext)


def average_hull_shape(trials_verts, n_angles=72):
    """Average several centered hull polygons (one per trial) into a single
    representative shape, by averaging their radius-vs-angle profiles
    rather than cherry-picking one trial."""
    query_angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    profiles = np.stack([radial_profile(v, query_angles) for v in trials_verts])
    r_mean = profiles.mean(axis=0)
    r_std = profiles.std(axis=0)
    mean_verts = np.column_stack([r_mean * np.cos(query_angles), r_mean * np.sin(query_angles)])
    return mean_verts, query_angles, r_mean, r_std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--label", default=None,
                     help="Restrict to one label, e.g. 'd'. Omit to aggregate ALL letters "
                          "per participant (recommended for an overall size comparison).")
    ap.add_argument("--participants", nargs="*", default=None)
    ap.add_argument("--source", choices=["pixel", "calibrated", "smoothed_mm"], default="smoothed_mm",
                     help="pixel = raw px from trajectory_smooth_120hz.csv (needs identical camera "
                          "framing across sessions); calibrated = local_x_mm/local_y_mm from the "
                          "raw (jittery) trajectory.csv; smoothed_mm (recommended) = the clean "
                          "smoothed trajectory rescaled to mm using a per-trial scale derived from "
                          "that trial's own calibrated trajectory.csv")
    ap.add_argument("--show-all-trials", action="store_true",
                     help="Draw every trial's hull outline per participant (unfilled), "
                          "instead of one filled representative shape each")
    ap.add_argument("--out", default="figure_size_variation.pdf")
    ap.add_argument("--also-png", action="store_true")
    args = ap.parse_args()

    set_paper_style()
    unit = "px" if args.source == "pixel" else "mm"

    trial_dirs = find_trial_dirs(args.dataset_root, args.label, args.participants)
    if not trial_dirs:
        scope = f"label='{args.label}'" if args.label else "any label"
        print(f"[WARN] No trials found for {scope}.")
        return

    per_participant_trials = {}  # participant -> list of (area, centered_hull_vertices)
    for trial_dir in trial_dirs:
        participant = parse_participant(trial_dir)
        xy = load_writing_xy(trial_dir, args.source)
        if xy is None:
            continue
        x, y = xy
        area, verts = hull_area_and_vertices(x, y)
        per_participant_trials.setdefault(participant, []).append((area, verts))

    if not per_participant_trials:
        print("[WARN] No usable trials (check --source and file availability).")
        return

    participants_all = sorted(per_participant_trials.keys())
    areas_per_p = {p: np.array([a for a, _ in per_participant_trials[p]]) for p in participants_all}
    means = np.array([areas_per_p[p].mean() for p in participants_all])
    stds = np.array([areas_per_p[p].std() for p in participants_all])
    ns = [len(per_participant_trials[p]) for p in participants_all]

    # average shape per participant: radial-profile average across ALL of
    # their trials, not a single cherry-picked one
    average_shapes = {}
    for p in participants_all:
        trials = per_participant_trials[p]
        verts_list = [v for _, v in trials]
        mean_verts, query_angles, r_mean, r_std = average_hull_shape(verts_list)
        average_shapes[p] = (mean_verts, query_angles, r_mean, r_std)

    order = np.argsort(-means)  # largest first, so smaller shapes draw on top and stay visible
    participants = [participants_all[i] for i in order]
    means, stds, ns = means[order], stds[order], [ns[i] for i in order]

    fig, (ax_shape, ax_legend, ax_bar) = plt.subplots(
        1, 3, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.25, 0.55, 1.1]})

    # (a) Overlaid OUTLINES (no fill) -- avoids the muddy color-blend that
    # semi-transparent fills produce once many shapes stack on top of each
    # other, while still showing nested size differences clearly.
    ax_shape.set_aspect("equal")
    max_extent = max(np.abs(average_shapes[p][0]).max() for p in participants) * 1.15
    if args.show_all_trials:
        for p in participants:
            for _, verts in per_participant_trials[p]:
                max_extent = max(max_extent, np.abs(verts).max() * 1.15)

    for i, p in enumerate(participants):
        color = PALETTE[i % len(PALETTE)]
        mean_verts, *_ = average_shapes[p]

        if args.show_all_trials:
            for _, verts in per_participant_trials[p]:
                ax_shape.add_patch(Polygon(verts, closed=True, facecolor="none",
                                            edgecolor=color, alpha=0.18, linewidth=0.5))

        ax_shape.add_patch(Polygon(mean_verts, closed=True, facecolor="none",
                                    edgecolor=color, alpha=0.9, linewidth=1.7))

    ax_shape.set_xlim(-max_extent, max_extent)
    ax_shape.set_ylim(max_extent, -max_extent)  # inverted (image/mm y-down convention)
    ax_shape.set_xlabel(f"x ({unit}, centered)", fontsize=8.5)
    ax_shape.set_ylabel(f"y ({unit}, centered)", fontsize=8.5)
    ax_shape.tick_params(labelsize=7)
    ax_shape.grid(alpha=0.12, linewidth=0.5)
    for spine in ax_shape.spines.values():
        spine.set_linewidth(0.6)
    subtitle = "mean shape outlines + all-trial outlines" if args.show_all_trials else "mean shape outlines (overlaid)"
    ax_shape.set_title(subtitle, fontsize=9)

    # compact legend list instead of one leader line per participant --
    # scales cleanly to many participants without visual clutter
    ax_legend.axis("off")
    legend_handles = [
        Line2D([0], [0], color=PALETTE[i % len(PALETTE)], lw=2,
               label=f"{p}  ({means[i]:.0f} {unit}\u00b2)")
        for i, p in enumerate(participants)
    ]
    ax_legend.legend(handles=legend_handles, loc="center left", fontsize=7.5,
                      frameon=False, handlelength=1.4, labelspacing=0.7)

    # (b) box plot: shows the actual per-trial distribution (median, IQR,
    # outliers) per participant, not just mean +/- SD
    x_pos = np.arange(len(participants))
    box_data = [areas_per_p[p] for p in participants]
    bp = ax_bar.boxplot(
        box_data, positions=x_pos, patch_artist=True, widths=0.6, showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="#333330", markersize=4),
        medianprops=dict(color="#333330", linewidth=1.3),
        whiskerprops=dict(color="#666660", linewidth=0.8),
        capprops=dict(color="#666660", linewidth=0.8),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="#888880",
                          markeredgecolor="none", alpha=0.6),
    )
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(PALETTE[i % len(PALETTE)])
        box.set_alpha(0.75)
        box.set_edgecolor(PALETTE[i % len(PALETTE)])

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(participants, rotation=45, ha="right", fontsize=7.5)
    ax_bar.set_ylabel(f"writing area ({unit}\u00b2)", fontsize=8.5)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for i, p in enumerate(participants):
        top = box_data[i].max()
        ax_bar.text(x_pos[i], top, f"n={ns[i]}", ha="center", va="bottom", fontsize=6)
    ax_bar.set_title("Writing area distribution per participant (box plot)", fontsize=9)

    scope_str = f"letter '{args.label}'" if args.label else "all letters (aggregated)"
    fig.suptitle(f"Writing size variation across participants \u2014 {scope_str}"
                 f"  ({unit}-based area)", fontsize=10)
    if not args.label:
        fig.text(0.5, 0.965,
                  "Shapes are an aggregate size/roundness envelope across all letters, "
                  "not any single letter's true shape.",
                  ha="center", fontsize=7, color="#666660")
    fig.tight_layout(rect=[0, 0, 1, 0.90 if not args.label else 0.93])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[DONE] Saved figure to {args.out}")
    if args.also_png:
        png_path = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(png_path, dpi=400, bbox_inches="tight")
        print(f"[DONE] Saved high-res PNG to {png_path}")

    print("\nPer-participant summary:")
    for p, m, s, n in zip(participants, means, stds, ns):
        print(f"  {p}: mean={m:.1f}{unit}^2  sd={s:.1f}{unit}^2  n={n}")


if __name__ == "__main__":
    main()