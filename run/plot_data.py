# """
# plot_multimodal_overview.py

# Plot all four sensing modalities for a single trial -- watch IMU, fingertip
# IMU, watch mic, and surface mic -- as a paper-style, time-aligned stacked
# figure, with touch-on/off spans shaded across every panel for reference.

# Column names for imu.csv / fingertip_imu.csv are auto-detected (accel/gyro,
# x/y/z, and finger name for the fingertip file). If detection fails, the
# script prints the actual columns found so you can pass overrides with
# --watch-accel-cols / --watch-gyro-cols / --finger-accel-cols / --finger-gyro-cols
# (comma-separated column names, e.g. "ax,ay,az").

# Usage:
#     # Point directly at a trial folder
#     python plot_multimodal_overview.py --trial-dir dataset/p1/dataset/d/trial_005 \
#         --out figure_multimodal_overview.pdf --also-png

#     # Or resolve by parts
#     python plot_multimodal_overview.py --dataset-root dataset --participant p1 \
#         --label d --trial trial_005 --out figure_multimodal_overview.pdf
# """
# import argparse
# import json
# import os
# import re

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy import signal
# from scipy.io import wavfile

# PALETTE = {
#     "watch_imu": "#378ADD",
#     "fingertip_imu": "#1D9E75",
#     "watch_mic": "#D85A30",
#     "surface_mic": "#BA7517",
# }


# def set_paper_style():
#     plt.rcParams.update({
#         "font.family": "serif",
#         "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
#         "font.size": 9,
#         "axes.linewidth": 0.6,
#         "pdf.fonttype": 42,
#         "ps.fonttype": 42,
#         "savefig.facecolor": "white",
#         "figure.facecolor": "white",
#     })


# def resolve_trial_dir(args):
#     if args.trial_dir:
#         return args.trial_dir
#     return os.path.join(args.dataset_root, args.participant, "dataset", args.label, args.trial)


# def find_col(columns, patterns):
#     for pat in patterns:
#         for c in columns:
#             if re.search(pat, c, re.IGNORECASE):
#                 return c
#     return None


# def load_long_format_signal_from_df(df, sensor_patterns, sensor_col="sensor",
#                                      value_cols=("v1", "v2", "v3"), time_col="time_aligned"):
#     if sensor_col not in df.columns:
#         return None, None
#     for pat in sensor_patterns:
#         mask = df[sensor_col].astype(str).str.contains(pat, case=False, regex=True, na=False)
#         if mask.any():
#             sub = df[mask]
#             t = sub[time_col].to_numpy(float) if time_col in sub.columns else np.arange(len(sub))
#             vals = sub[list(value_cols)].to_numpy(float)
#             return t, vals
#     print(f"[WARN] No rows matched sensor patterns {sensor_patterns} (column '{sensor_col}'). "
#           f"Unique values found: {sorted(df[sensor_col].astype(str).unique())}")
#     return None, None


# def load_long_format_signal(csv_path, sensor_patterns, sensor_col="sensor",
#                              value_cols=("v1", "v2", "v3"), time_col="time_aligned"):
#     """Load a signal stored in long format: one row per (time, sensor_type),
#     with the x/y/z (or similar) values in value_cols. sensor_patterns are
#     tried in order as case-insensitive substring/regex matches against the
#     sensor_col values."""
#     df = pd.read_csv(csv_path)
#     return load_long_format_signal_from_df(df, sensor_patterns, sensor_col, value_cols, time_col)


# def detect_axes(df, prefix_patterns, override=None):
#     """Find x/y/z column names for a signal (e.g. accel or gyro), either
#     from an explicit comma-separated override or by regex auto-detection."""
#     if override:
#         cols = [c.strip() for c in override.split(",")]
#         missing = [c for c in cols if c not in df.columns]
#         if missing:
#             raise ValueError(f"Override columns not found: {missing}. Available: {list(df.columns)}")
#         return cols

#     cols = list(df.columns)
#     axes = []
#     for axis in ("x", "y", "z"):
#         pats = [p.format(axis=axis) for p in prefix_patterns]
#         col = find_col(cols, pats)
#         axes.append(col)
#     if None in axes:
#         return None
#     return axes


# def load_watch_imu(csv_path, accel_override=None, gyro_override=None):
#     df = pd.read_csv(csv_path)

#     # Try long format first: time_aligned, sensor, v1, v2, v3
#     if "sensor" in df.columns and not (accel_override or gyro_override):
#         t_a, accel = load_long_format_signal(csv_path, [r"acc"])
#         t_g, gyro = load_long_format_signal(csv_path, [r"gyro"])
#         if accel is not None or gyro is not None:
#             t = t_a if t_a is not None else t_g
#             return t, accel, gyro

#     # Fall back to wide format: separate accel_x/accel_y/accel_z-style columns
#     t_col = find_col(df.columns, [r"^time_aligned$", r"time"]) or df.columns[0]
#     t = df[t_col].to_numpy(float)

#     accel_cols = detect_axes(df, [r"acc.*_?{axis}$", r"^{axis}$"], accel_override)
#     gyro_cols = detect_axes(df, [r"gyro.*_?{axis}$"], gyro_override)

#     if accel_cols is None:
#         print(f"[WARN] Could not auto-detect accel columns in {csv_path}. "
#               f"Available columns: {list(df.columns)}. Use --watch-accel-cols to override.")
#     if gyro_cols is None:
#         print(f"[WARN] Could not auto-detect gyro columns in {csv_path}. "
#               f"Available columns: {list(df.columns)}. Use --watch-gyro-cols to override.")

#     accel = df[accel_cols].to_numpy(float) if accel_cols else None
#     gyro = df[gyro_cols].to_numpy(float) if gyro_cols else None
#     return t, accel, gyro


# def load_fingertip_imu(csv_path, finger="index", accel_override=None, gyro_override=None,
#                         only_detected=True):
#     """fingertip_imu.csv is one row per (time, finger): a 'finger' column
#     (thumb/index/middle/ring/pinky) selects the row subset, and that subset
#     already has wide-format accel_x/accel_y/accel_z, gyro_x/gyro_y/gyro_z
#     columns -- no long-format pivot needed here."""
#     df = pd.read_csv(csv_path)

#     if "finger" in df.columns:
#         mask = df["finger"].astype(str).str.lower() == finger.lower()
#         if not mask.any():
#             print(f"[WARN] No rows matched finger='{finger}' in {csv_path}. "
#                   f"Unique values: {sorted(df['finger'].astype(str).unique())}")
#         else:
#             df = df[mask]
#     else:
#         print(f"[WARN] No 'finger' column in {csv_path}; using all rows. "
#               f"Available columns: {list(df.columns)}")

#     if only_detected and "detected" in df.columns:
#         df = df[df["detected"] == 1]

#     t_col = find_col(df.columns, [r"^time_aligned$", r"time"]) or df.columns[0]
#     t = df[t_col].to_numpy(float)

#     accel_cols = detect_axes(df, [r"accel_?{axis}$", r"acc.*_?{axis}$"], accel_override)
#     gyro_cols = detect_axes(df, [r"gyro_?{axis}$"], gyro_override)

#     if accel_cols is None:
#         print(f"[WARN] Could not find fingertip accel columns in {csv_path}. "
#               f"Available: {list(df.columns)}. Use --finger-accel-cols to override.")
#     if gyro_cols is None:
#         print(f"[WARN] Could not find fingertip gyro columns in {csv_path}. "
#               f"Available: {list(df.columns)}. Use --finger-gyro-cols to override.")

#     accel = df[accel_cols].to_numpy(float) if accel_cols else None
#     gyro = df[gyro_cols].to_numpy(float) if gyro_cols else None
#     return t, accel, gyro


# def load_audio(wav_path):
#     sr, data = wavfile.read(wav_path)
#     if data.ndim > 1:
#         data = data.mean(axis=1)
#     if np.issubdtype(data.dtype, np.integer):
#         data = data.astype(np.float32) / np.iinfo(data.dtype).max
#     t = np.arange(len(data)) / sr
#     return t, data, sr


# def load_touch_spans(trial_dir):
#     events_path = os.path.join(trial_dir, "events.csv")
#     if not os.path.exists(events_path):
#         return []
#     ev = pd.read_csv(events_path)
#     spans, on_time = [], None
#     for _, row in ev.sort_values("time_aligned").iterrows():
#         if row["event"] == "audio_touch_on":
#             on_time = row["time_aligned"]
#         elif row["event"] == "audio_touch_off" and on_time is not None:
#             spans.append((on_time, row["time_aligned"]))
#             on_time = None
#     return spans


# def plot_imu_panel(ax, t, accel, gyro, color, label):
#     axis_labels = ["x", "y", "z"]
#     linestyles = ["-", "--", ":"]
#     if accel is not None and t is not None:
#         n_axes = accel.shape[1] if accel.ndim == 2 else 1
#         accel_2d = accel if accel.ndim == 2 else accel.reshape(-1, 1)
#         for i in range(n_axes):
#             ax.plot(t, accel_2d[:, i], color=color, linestyle=linestyles[min(i, 2)],
#                      linewidth=1.1, label=axis_labels[min(i, 2)])
#         ax.legend(loc="upper right", fontsize=6, frameon=False, ncol=3,
#                    handlelength=1.5, columnspacing=0.8)
#     else:
#         ax.text(0.5, 0.5, "no accel data found", transform=ax.transAxes,
#                  ha="center", va="center", fontsize=7, color="red")
#     ax.set_ylabel(f"{label}\naccel", fontsize=8)
#     ax.tick_params(labelsize=7)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)


# def plot_audio_waveform(ax, t, wave, color, label):
#     ax.plot(t, wave, color=color, linewidth=0.4)
#     ax.set_ylabel(label, fontsize=8)
#     ax.set_yticks([])
#     ax.tick_params(labelsize=7)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["left"].set_visible(False)


# def plot_spectrogram(fig, ax, wave, sr, nperseg=256, cmap="magma"):
#     noverlap = nperseg // 2
#     f, t_spec, Sxx = signal.spectrogram(wave, fs=sr, nperseg=nperseg, noverlap=noverlap)
#     Sxx_db = 10 * np.log10(Sxx + 1e-10)
#     im = ax.pcolormesh(t_spec, f / 1000, Sxx_db, shading="auto", cmap=cmap)
#     ax.set_ylabel("freq (kHz)", fontsize=8)
#     ax.tick_params(labelsize=7)
#     for spine in ax.spines.values():
#         spine.set_visible(False)
#     cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.008)
#     cbar.ax.tick_params(labelsize=6)
#     cbar.set_label("dB", fontsize=6)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--trial-dir", default=None, help="Direct path to a trial folder")
#     ap.add_argument("--dataset-root")
#     ap.add_argument("--participant")
#     ap.add_argument("--label")
#     ap.add_argument("--trial")
#     ap.add_argument("--finger", default="index")
#     ap.add_argument("--watch-accel-cols", default=None, help="Override, e.g. 'ax,ay,az'")
#     ap.add_argument("--watch-gyro-cols", default=None)
#     ap.add_argument("--finger-accel-cols", default=None)
#     ap.add_argument("--finger-gyro-cols", default=None)
#     ap.add_argument("--out", default="figure_multimodal_overview.pdf")
#     ap.add_argument("--also-png", action="store_true")
#     args = ap.parse_args()

#     trial_dir = resolve_trial_dir(args)
#     if not os.path.isdir(trial_dir):
#         raise SystemExit(f"Trial folder not found: {trial_dir}")

#     set_paper_style()

#     t_wimu, accel_w, gyro_w = load_watch_imu(
#         os.path.join(trial_dir, "imu.csv"), args.watch_accel_cols, args.watch_gyro_cols)
#     t_fimu, accel_f, gyro_f = load_fingertip_imu(
#         os.path.join(trial_dir, "fingertip_imu.csv"), args.finger,
#         args.finger_accel_cols, args.finger_gyro_cols)
#     t_wmic, wave_wmic, sr_w = load_audio(os.path.join(trial_dir, "watch_audio.wav"))
#     t_smic, wave_smic, sr_s = load_audio(os.path.join(trial_dir, "surface_mic.wav"))
#     spans = load_touch_spans(trial_dir)

#     height_ratios = [1, 1, 0.45, 1.2, 0.45, 1.2]
#     fig = plt.figure(figsize=(7.4, 9.5))
#     gs = fig.add_gridspec(6, 1, height_ratios=height_ratios, hspace=0.25)

#     ax_wimu = fig.add_subplot(gs[0])
#     ax_fimu = fig.add_subplot(gs[1], sharex=ax_wimu)
#     ax_wmic_wave = fig.add_subplot(gs[2], sharex=ax_wimu)
#     ax_wmic_spec = fig.add_subplot(gs[3], sharex=ax_wimu)
#     ax_smic_wave = fig.add_subplot(gs[4], sharex=ax_wimu)
#     ax_smic_spec = fig.add_subplot(gs[5], sharex=ax_wimu)
#     axes = [ax_wimu, ax_fimu, ax_wmic_wave, ax_wmic_spec, ax_smic_wave, ax_smic_spec]

#     plot_imu_panel(ax_wimu, t_wimu, accel_w, gyro_w, PALETTE["watch_imu"], "watch IMU")
#     plot_imu_panel(ax_fimu, t_fimu, accel_f, gyro_f, PALETTE["fingertip_imu"], f"fingertip IMU ({args.finger})")
#     plot_audio_waveform(ax_wmic_wave, t_wmic, wave_wmic, PALETTE["watch_mic"], "watch mic")
#     plot_spectrogram(fig, ax_wmic_spec, wave_wmic, sr_w)
#     plot_audio_waveform(ax_smic_wave, t_smic, wave_smic, PALETTE["surface_mic"], "surface mic")
#     plot_spectrogram(fig, ax_smic_spec, wave_smic, sr_s)

#     for ax in axes:
#         for on_t, off_t in spans:
#             ax.axvspan(on_t, off_t, color="gray", alpha=0.12, linewidth=0)
#         if ax is not axes[-1]:
#             plt.setp(ax.get_xticklabels(), visible=False)

#     axes[-1].set_xlabel("time (s)", fontsize=9)
#     participant = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(trial_dir))))
#     label = os.path.basename(os.path.dirname(trial_dir))
#     trial = os.path.basename(trial_dir)
#     fig.suptitle(f"Multimodal signal overview \u2014 {participant} / '{label}' / {trial}"
#                  f"  (shaded = touch-on/off, from events.csv)", fontsize=10)

#     fig.tight_layout(rect=[0, 0, 1, 0.97])
#     fig.savefig(args.out, bbox_inches="tight")
#     print(f"[DONE] Saved figure to {args.out}")
#     if args.also_png:
#         png_path = os.path.splitext(args.out)[0] + ".png"
#         fig.savefig(png_path, dpi=400, bbox_inches="tight")
#         print(f"[DONE] Saved high-res PNG to {png_path}")


# if __name__ == "__main__":
#     main()

"""
plot_multimodal_overview.py

Plot all four sensing modalities for a single trial -- watch IMU, fingertip
IMU, watch mic, and surface mic -- as a paper-style, time-aligned stacked
figure, with touch-on/off spans shaded across every panel for reference.

Fingertip IMU is shown TWICE: once as recorded (raw, straight off
MediaPipe's webcam hand tracking) and once after the same two-stage
denoising this project's training pipeline applies before using this
signal as either a model input or a teacher target (see dataset_ctc.py's
_denoise_fingertip_signal) -- median filter (reject single-frame
occlusion/tracking spikes) then Savitzky-Golay (smooth while preserving
real motion shape). Reimplemented standalone here (not imported from
dataset_ctc.py) so this plotting script keeps its own lightweight
dependencies (matplotlib/numpy/pandas/scipy only) and doesn't need
torch/config_ctc just to draw a figure.

Column names for imu.csv / fingertip_imu.csv are auto-detected (accel/gyro,
x/y/z, and finger name for the fingertip file). If detection fails, the
script prints the actual columns found so you can pass overrides with
--watch-accel-cols / --watch-gyro-cols / --finger-accel-cols / --finger-gyro-cols
(comma-separated column names, e.g. "ax,ay,az").

Usage:
    # Point directly at a trial folder
    python plot_multimodal_overview.py --trial-dir dataset/p1/dataset/d/trial_005 \
        --out figure_multimodal_overview.pdf --also-png

    # Or resolve by parts
    python plot_multimodal_overview.py --dataset-root dataset --participant p1 \
        --label d --trial trial_005 --out figure_multimodal_overview.pdf
"""
import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

PALETTE = {
    "watch_imu": "#378ADD",
    "fingertip_imu": "#1D9E75",
    "fingertip_imu_denoised": "#0B6E4F",
    "watch_mic": "#D85A30",
    "surface_mic": "#BA7517",
}


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


def find_col(columns, patterns):
    for pat in patterns:
        for c in columns:
            if re.search(pat, c, re.IGNORECASE):
                return c
    return None


def denoise_fingertip_signal(values):
    """(T, C) -> (T, C) -- the SAME two-stage denoising the training
    pipeline applies to fingertip IMU (see dataset_ctc.py's
    _denoise_fingertip_signal), reimplemented standalone here against
    this script's own (T, C) time-major array convention (the training
    pipeline uses (C, T) channel-major, since it works with torch
    tensors -- purely a shape-convention difference, same math).

    1) Median filter (kernel=5, ~125ms at a typical webcam frame rate)
       rejects single-frame spikes -- a momentary bad MediaPipe
       detection (e.g. brief occlusion) jumping to a wildly wrong
       value -- which a plain moving average would only blend into
       its neighbors, not remove.
    2) Savitzky-Golay (window=9, 2nd-order polynomial) smooths the
       continuous low-level frame-to-frame jitter that remains,
       preserving the real SHAPE of genuine finger motion (peaks,
       slopes, curvature) far better than a plain moving average does.

    Returns the input unchanged if there are too few samples (<5) for
    either filter to do anything meaningful, or if values is None
    (mirrors the training pipeline's own short-trial no-op behavior)."""
    if values is None or len(values) < 5:
        return values
    T = values.shape[0]
    median_kernel = min(5, T if T % 2 == 1 else T - 1)
    if median_kernel >= 3:
        despiked = np.stack(
            [signal.medfilt(values[:, c], kernel_size=median_kernel) for c in range(values.shape[1])],
            axis=1)
    else:
        despiked = values
    savgol_window = min(9, T if T % 2 == 1 else T - 1)
    if savgol_window > 3:
        smoothed = signal.savgol_filter(despiked, window_length=savgol_window, polyorder=2, axis=0)
    else:
        smoothed = despiked
    return smoothed


def load_long_format_signal_from_df(df, sensor_patterns, sensor_col="sensor",
                                     value_cols=("v1", "v2", "v3"), time_col="time_aligned"):
    if sensor_col not in df.columns:
        return None, None
    for pat in sensor_patterns:
        mask = df[sensor_col].astype(str).str.contains(pat, case=False, regex=True, na=False)
        if mask.any():
            sub = df[mask]
            t = sub[time_col].to_numpy(float) if time_col in sub.columns else np.arange(len(sub))
            vals = sub[list(value_cols)].to_numpy(float)
            return t, vals
    print(f"[WARN] No rows matched sensor patterns {sensor_patterns} (column '{sensor_col}'). "
          f"Unique values found: {sorted(df[sensor_col].astype(str).unique())}")
    return None, None


def load_long_format_signal(csv_path, sensor_patterns, sensor_col="sensor",
                             value_cols=("v1", "v2", "v3"), time_col="time_aligned"):
    """Load a signal stored in long format: one row per (time, sensor_type),
    with the x/y/z (or similar) values in value_cols. sensor_patterns are
    tried in order as case-insensitive substring/regex matches against the
    sensor_col values."""
    df = pd.read_csv(csv_path)
    return load_long_format_signal_from_df(df, sensor_patterns, sensor_col, value_cols, time_col)


def detect_axes(df, prefix_patterns, override=None):
    """Find x/y/z column names for a signal (e.g. accel or gyro), either
    from an explicit comma-separated override or by regex auto-detection."""
    if override:
        cols = [c.strip() for c in override.split(",")]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Override columns not found: {missing}. Available: {list(df.columns)}")
        return cols

    cols = list(df.columns)
    axes = []
    for axis in ("x", "y", "z"):
        pats = [p.format(axis=axis) for p in prefix_patterns]
        col = find_col(cols, pats)
        axes.append(col)
    if None in axes:
        return None
    return axes


def load_watch_imu(csv_path, accel_override=None, gyro_override=None):
    df = pd.read_csv(csv_path)

    # Try long format first: time_aligned, sensor, v1, v2, v3
    if "sensor" in df.columns and not (accel_override or gyro_override):
        t_a, accel = load_long_format_signal(csv_path, [r"acc"])
        t_g, gyro = load_long_format_signal(csv_path, [r"gyro"])
        if accel is not None or gyro is not None:
            t = t_a if t_a is not None else t_g
            return t, accel, gyro

    # Fall back to wide format: separate accel_x/accel_y/accel_z-style columns
    t_col = find_col(df.columns, [r"^time_aligned$", r"time"]) or df.columns[0]
    t = df[t_col].to_numpy(float)

    accel_cols = detect_axes(df, [r"acc.*_?{axis}$", r"^{axis}$"], accel_override)
    gyro_cols = detect_axes(df, [r"gyro.*_?{axis}$"], gyro_override)

    if accel_cols is None:
        print(f"[WARN] Could not auto-detect accel columns in {csv_path}. "
              f"Available columns: {list(df.columns)}. Use --watch-accel-cols to override.")
    if gyro_cols is None:
        print(f"[WARN] Could not auto-detect gyro columns in {csv_path}. "
              f"Available columns: {list(df.columns)}. Use --watch-gyro-cols to override.")

    accel = df[accel_cols].to_numpy(float) if accel_cols else None
    gyro = df[gyro_cols].to_numpy(float) if gyro_cols else None
    return t, accel, gyro


def load_fingertip_imu(csv_path, finger="index", accel_override=None, gyro_override=None,
                        only_detected=True):
    """fingertip_imu.csv is one row per (time, finger): a 'finger' column
    (thumb/index/middle/ring/pinky) selects the row subset, and that subset
    already has wide-format accel_x/accel_y/accel_z, gyro_x/gyro_y/gyro_z
    columns -- no long-format pivot needed here."""
    df = pd.read_csv(csv_path)

    if "finger" in df.columns:
        mask = df["finger"].astype(str).str.lower() == finger.lower()
        if not mask.any():
            print(f"[WARN] No rows matched finger='{finger}' in {csv_path}. "
                  f"Unique values: {sorted(df['finger'].astype(str).unique())}")
        else:
            df = df[mask]
    else:
        print(f"[WARN] No 'finger' column in {csv_path}; using all rows. "
              f"Available columns: {list(df.columns)}")

    if only_detected and "detected" in df.columns:
        df = df[df["detected"] == 1]

    t_col = find_col(df.columns, [r"^time_aligned$", r"time"]) or df.columns[0]
    t = df[t_col].to_numpy(float)

    accel_cols = detect_axes(df, [r"accel_?{axis}$", r"acc.*_?{axis}$"], accel_override)
    gyro_cols = detect_axes(df, [r"gyro_?{axis}$"], gyro_override)

    if accel_cols is None:
        print(f"[WARN] Could not find fingertip accel columns in {csv_path}. "
              f"Available: {list(df.columns)}. Use --finger-accel-cols to override.")
    if gyro_cols is None:
        print(f"[WARN] Could not find fingertip gyro columns in {csv_path}. "
              f"Available: {list(df.columns)}. Use --finger-gyro-cols to override.")

    accel = df[accel_cols].to_numpy(float) if accel_cols else None
    gyro = df[gyro_cols].to_numpy(float) if gyro_cols else None
    return t, accel, gyro


def load_audio(wav_path):
    sr, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    t = np.arange(len(data)) / sr
    return t, data, sr


def load_touch_spans(trial_dir):
    events_path = os.path.join(trial_dir, "events.csv")
    if not os.path.exists(events_path):
        return []
    ev = pd.read_csv(events_path)
    spans, on_time = [], None
    for _, row in ev.sort_values("time_aligned").iterrows():
        if row["event"] == "audio_touch_on":
            on_time = row["time_aligned"]
        elif row["event"] == "audio_touch_off" and on_time is not None:
            spans.append((on_time, row["time_aligned"]))
            on_time = None
    return spans


def plot_imu_panel(ax, t, accel, gyro, color, label):
    axis_labels = ["x", "y", "z"]

    # x, y, z 축을 각각 빨강, 초록, 파랑 계열의 명확한 색상으로 지정 (원하시는 헥스코드로 변경 가능)
    axis_colors = ["#e41a1c", "#4daf4a", "#377eb8"]

    if accel is not None and t is not None:
        n_axes = accel.shape[1] if accel.ndim == 2 else 1
        accel_2d = accel if accel.ndim == 2 else accel.reshape(-1, 1)
        for i in range(n_axes):
            # linestyle을 "-" (실선)으로 고정하고, color를 axis_colors에서 가져옴
            ax.plot(t, accel_2d[:, i], color=axis_colors[min(i, 2)], linestyle="-",
                     linewidth=1.1, label=axis_labels[min(i, 2)])

        ax.legend(loc="upper right", fontsize=6, frameon=False, ncol=3,
                   handlelength=1.5, columnspacing=0.8)
    else:
        ax.text(0.5, 0.5, "no accel data found", transform=ax.transAxes,
                 ha="center", va="center", fontsize=7, color="red")

    ax.set_ylabel(f"{label}\naccel", fontsize=7, labelpad=2)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_audio_waveform(ax, t, wave, color, label):
    ax.plot(t, wave, color=color, linewidth=0.4)
    ax.set_ylabel(label, fontsize=8)
    ax.set_yticks([])
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def plot_spectrogram(fig, ax, wave, sr, nperseg=1024, cmap="magma", denoise=True):
    # 1. High-pass Filter: 1kHz 이하의 저주파 노이즈(움직임, 울림) 제거
    if denoise:
        # 4차 Butterworth 고대역 통과 필터 생성 (컷오프 1000Hz)
        sos = signal.butter(4, 1000, 'hp', fs=sr, output='sos')
        wave = signal.sosfiltfilt(sos, wave)  # 위상 지연이 없는 양방향 필터링

    noverlap = nperseg // 2
    f, t_spec, Sxx = signal.spectrogram(wave, fs=sr, nperseg=nperseg, noverlap=noverlap)

    # 2. Spectral Median Subtraction: 지속적인 배경 화이트 노이즈 제거
    if denoise:
        # 시간 축(axis=1)을 기준으로 각 주파수별 중간값을 구함 (노이즈 프로필)
        noise_profile = np.median(Sxx, axis=1, keepdims=True)
        # 전체 스펙트로그램에서 노이즈 프로필을 빼고, 음수가 된 값은 아주 작은 값(1e-10)으로 보정
        Sxx = np.clip(Sxx - noise_profile, 1e-10, None)

    # 데시벨(dB) 변환
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    # 시각화
    im = ax.pcolormesh(t_spec, f / 1000, Sxx_db, shading="auto", cmap=cmap)

    ax.set_ylabel("freq (kHz)", fontsize=8)
    ax.set_ylim(0, 16)  # 마찰음 주요 대역에 집중
    ax.tick_params(labelsize=7)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.008)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("dB", fontsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-dir", default=None, help="Direct path to a trial folder")
    ap.add_argument("--dataset-root")
    ap.add_argument("--participant")
    ap.add_argument("--label")
    ap.add_argument("--trial")
    ap.add_argument("--finger", default="index")
    ap.add_argument("--watch-accel-cols", default=None, help="Override, e.g. 'ax,ay,az'")
    ap.add_argument("--watch-gyro-cols", default=None)
    ap.add_argument("--finger-accel-cols", default=None)
    ap.add_argument("--finger-gyro-cols", default=None)
    ap.add_argument("--out", default="figure_multimodal_overview.pdf")
    ap.add_argument("--also-png", action="store_true")
    ap.add_argument("--no-denoise-panel", action="store_true",
                     help="skip the extra denoised-fingertip-IMU panel and reproduce the original "
                          "(pre-denoising) figure layout exactly")
    args = ap.parse_args()

    trial_dir = resolve_trial_dir(args)
    if not os.path.isdir(trial_dir):
        raise SystemExit(f"Trial folder not found: {trial_dir}")

    set_paper_style()

    t_wimu, accel_w, gyro_w = load_watch_imu(
        os.path.join(trial_dir, "imu.csv"), args.watch_accel_cols, args.watch_gyro_cols)
    t_fimu, accel_f, gyro_f = load_fingertip_imu(
        os.path.join(trial_dir, "fingertip_imu.csv"), args.finger,
        args.finger_accel_cols, args.finger_gyro_cols)
    t_wmic, wave_wmic, sr_w = load_audio(os.path.join(trial_dir, "watch_audio.wav"))
    t_smic, wave_smic, sr_s = load_audio(os.path.join(trial_dir, "surface_mic.wav"))
    spans = load_touch_spans(trial_dir)

    show_denoise_panel = accel_f is not None and not args.no_denoise_panel
    if show_denoise_panel:
        accel_f_denoised = denoise_fingertip_signal(accel_f)

    if show_denoise_panel:
        height_ratios = [1, 1, 1, 0.45, 1.2, 0.45, 1.2]
    else:
        height_ratios = [1, 1, 0.45, 1.2, 0.45, 1.2]
    n_rows = len(height_ratios)
    fig = plt.figure(figsize=(7.4, 9.5 + (1.1 if show_denoise_panel else 0)))
    gs = fig.add_gridspec(n_rows, 1, height_ratios=height_ratios, hspace=0.25)

    ax_wimu = fig.add_subplot(gs[0])
    ax_fimu = fig.add_subplot(gs[1], sharex=ax_wimu)
    row = 2
    if show_denoise_panel:
        ax_fimu_denoised = fig.add_subplot(gs[row], sharex=ax_wimu)
        row += 1
    ax_wmic_wave = fig.add_subplot(gs[row], sharex=ax_wimu); row += 1
    ax_wmic_spec = fig.add_subplot(gs[row], sharex=ax_wimu); row += 1
    ax_smic_wave = fig.add_subplot(gs[row], sharex=ax_wimu); row += 1
    ax_smic_spec = fig.add_subplot(gs[row], sharex=ax_wimu); row += 1
    axes = [ax_wimu, ax_fimu] + ([ax_fimu_denoised] if show_denoise_panel else []) + \
           [ax_wmic_wave, ax_wmic_spec, ax_smic_wave, ax_smic_spec]

    plot_imu_panel(ax_wimu, t_wimu, accel_w, gyro_w, PALETTE["watch_imu"], "watch IMU")
    plot_imu_panel(ax_fimu, t_fimu, accel_f, gyro_f, PALETTE["fingertip_imu"], f"fingertip IMU ({args.finger}, raw)")
    if show_denoise_panel:
        # gyro is intentionally omitted here (plot_imu_panel never actually
        # renders gyro even when passed one -- matches the raw panel above,
        # which has the same quirk) -- only accel is denoised/shown.
        plot_imu_panel(ax_fimu_denoised, t_fimu, accel_f_denoised, None,
                        PALETTE["fingertip_imu_denoised"], f"fingertip IMU ({args.finger}, denoised)")
    plot_audio_waveform(ax_wmic_wave, t_wmic, wave_wmic, PALETTE["watch_mic"], "watch mic")
    plot_spectrogram(fig, ax_wmic_spec, wave_wmic, sr_w)
    plot_audio_waveform(ax_smic_wave, t_smic, wave_smic, PALETTE["surface_mic"], "surface mic")
    plot_spectrogram(fig, ax_smic_spec, wave_smic, sr_s)

    for ax in axes:
        for on_t, off_t in spans:
            ax.axvspan(on_t, off_t, color="gray", alpha=0.12, linewidth=0)
        if ax is not axes[-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

    axes[-1].set_xlabel("time (s)", fontsize=9)
    participant = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(trial_dir))))
    label = os.path.basename(os.path.dirname(trial_dir))
    trial = os.path.basename(trial_dir)
    fig.suptitle(f"Multimodal signal overview \u2014 {participant} / '{label}' / {trial}"
                 f"  (shaded = touch-on/off, from events.csv)", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[DONE] Saved figure to {args.out}")
    if args.also_png:
        png_path = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(png_path, dpi=400, bbox_inches="tight")
        print(f"[DONE] Saved high-res PNG to {png_path}")


if __name__ == "__main__":
    main()