"""
select_trim_range.py

Interactively select the writing start/end time (and, for multi-stroke
letters, multiple separate ranges) for a single trial -- useful when
automatic touch-on/off detection from events.csv is unreliable, or when
movement before writing actually begins makes the raw trajectory hard to
read.

The LEFT plot shows time vs. y-position: drag across it to preview a range.
The RIGHT plot shows the full x-y trajectory, with the currently previewed
range highlighted in orange.

Keys:
    a  - append the current dragged selection as a saved range
    z  - remove the last appended range
    s  - save all appended ranges to manual_range.json in the trial folder
    q  - quit

Once manual_range.json exists for a trial, browse_trajectories.py and
plot_selected_trajectories.py will use it INSTEAD of events.csv for that
trial.

Usage:
    python select_trim_range.py --dataset-root dataset --label d --participant p1 --trial trial_001
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import SpanSelector

PREFERRED_TRAJ_FILENAMES = ["trajectory_smooth.csv", "trajectory.csv"]
PREFERRED_COORD_COLS = [
    ("local_x_mm", "local_y_mm"),
    ("pos_x_mm", "pos_y_mm"),
    ("x_px", "y_px"),
]


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
        raise ValueError(f"No known coordinate columns in {csv_path}")
    t_col = "time_aligned" if "time_aligned" in df.columns else df.columns[0]
    return df[t_col].to_numpy(float), df[x_col].to_numpy(float), df[y_col].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--trial", required=True)
    args = ap.parse_args()

    trial_dir = os.path.join(args.dataset_root, args.participant, "dataset", args.label, args.trial)
    traj_path = resolve_traj_path(trial_dir)
    t, x, y = load_trajectory(traj_path)

    manual_path = os.path.join(trial_dir, "manual_range.json")
    selected_ranges = []
    if os.path.exists(manual_path):
        with open(manual_path) as f:
            selected_ranges = [tuple(r) for r in json.load(f).get("ranges", [])]
        print(f"[INFO] Loaded {len(selected_ranges)} existing range(s) from {manual_path}")

    current = {"start": None, "end": None}

    fig, (ax_ts, ax_xy) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_ts.plot(t, y, color="#378ADD", linewidth=1)
    ax_ts.set_title("Drag to preview a range (time vs. y)")
    ax_ts.set_xlabel("time (s)")

    ax_xy.plot(x, y, color="#B0AFA8", linewidth=1)
    ax_xy.invert_yaxis()
    ax_xy.set_aspect("equal")
    title = ax_xy.set_title(f"{args.participant}/{args.trial}  ({len(selected_ranges)} range(s) saved)")
    highlight_line, = ax_xy.plot([], [], color="#D85A30", linewidth=2)

    def on_select(tmin, tmax):
        current["start"], current["end"] = tmin, tmax
        mask = (t >= tmin) & (t <= tmax)
        highlight_line.set_data(x[mask], y[mask])
        fig.canvas.draw_idle()

    span = SpanSelector(ax_ts, on_select, "horizontal", useblit=True,
                         props=dict(alpha=0.3, facecolor="#D85A30"))

    def refresh_title():
        title.set_text(f"{args.participant}/{args.trial}  ({len(selected_ranges)} range(s) saved)")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "a" and current["start"] is not None:
            selected_ranges.append((current["start"], current["end"]))
            print(f"[+] Added range #{len(selected_ranges)}: "
                  f"{current['start']:.2f}s - {current['end']:.2f}s")
            refresh_title()
        elif event.key == "z" and selected_ranges:
            removed = selected_ranges.pop()
            print(f"[-] Removed range: {removed[0]:.2f}s - {removed[1]:.2f}s")
            refresh_title()
        elif event.key == "s":
            if not selected_ranges:
                print("[WARN] No ranges added yet -- drag a selection, then press 'a' first.")
                return
            with open(manual_path, "w") as f:
                json.dump({"ranges": selected_ranges}, f, indent=2)
            print(f"[SAVED] {manual_path}: {selected_ranges}")
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Drag on the LEFT plot to preview a range (highlighted on the right).")
    print("Keys: 'a' add range, 'z' undo last, 's' save all, 'q' quit.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
