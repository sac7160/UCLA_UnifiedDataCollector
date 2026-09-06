"""
plot_audio_preprocessing_pipeline.py

Visualize how one audio channel (watch mic or surface mic) changes through
each preprocessing stage, using real data from one trial:

    (1) Raw waveform, full trial (as recorded)
    (2) Trimmed to [first touch-on, last touch-off] from events.csv
    (3) Resampled to 16 kHz
    (4) Log-mel spectrogram (n_fft=1024, hop_length=256, n_mels=64)
    (5) Z-score normalized log-mel spectrogram

No librosa dependency -- resampling uses scipy's polyphase resampler
(scipy.signal.resample_poly), and the mel filterbank / STFT / log-mel
spectrogram are implemented directly with scipy.signal.stft + numpy
(standard HTK-style mel scale: mel = 2595*log10(1 + f/700)).

Usage:
    python plot_audio_preprocessing_pipeline.py --trial-dir dataset/p1/dataset/d/trial_005 \
        --mic surface --out figure_audio_pipeline.pdf --also-png
"""
import argparse
import os
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly, stft

PANEL_EDGE = "#B0AFA8"
WAVE_COLOR = {"watch": "#D85A30", "surface": "#BA7517"}


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


def load_wav(path):
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)
    return sr, data


def get_trim_range(events_path):
    """First touch-on to last touch-off across the whole trial (a single
    span, not per-stroke pairs)."""
    ev = pd.read_csv(events_path)
    on_times = ev.loc[ev["event"] == "audio_touch_on", "time_aligned"]
    off_times = ev.loc[ev["event"] == "audio_touch_off", "time_aligned"]
    if on_times.empty or off_times.empty:
        return None
    return float(on_times.min()), float(off_times.max())


def resample_audio(wave, orig_sr, target_sr):
    """Polyphase resampling (scipy), no librosa needed."""
    frac = Fraction(target_sr, orig_sr).limit_denominator(1000)
    return resample_poly(wave, frac.numerator, frac.denominator).astype(np.float32)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Standard triangular mel filterbank (HTK-style mel scale)."""
    fmax = fmax or sr / 2
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    n_freq = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freq))
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, min(center, n_freq)):
            if center > left:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, min(right, n_freq)):
            if right > center:
                fb[m - 1, k] = (right - k) / (right - center)
    return fb, hz_pts[1:-1]  # filterbank, mel-bin center frequencies (Hz)


def log_mel_spectrogram(wave, sr, n_fft, hop_length, n_mels):
    f, t, Zxx = stft(wave, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length,
                       window="hann", boundary=None, padded=False)
    power = np.abs(Zxx) ** 2
    fb, mel_freqs = mel_filterbank(sr, n_fft, n_mels)
    mel_power = fb @ power
    mel_db = 10.0 * np.log10(np.maximum(mel_power, 1e-10))
    mel_db = mel_db - mel_db.max()  # reference to peak, like librosa's ref=np.max
    return mel_db, t, mel_freqs


def plot_waveform(ax, wave, sr, color, title, trim_lines=None):
    t = np.arange(len(wave)) / sr
    ax.plot(t, wave, color=color, linewidth=0.5)
    if trim_lines:
        for x in trim_lines:
            ax.axvline(x, color="#333330", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("amp.", fontsize=8)
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.tick_params(labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_spectrogram(fig, ax, mel_db, t, mel_freqs, title, cmap="magma", cbar_label="dB"):
    im = ax.pcolormesh(t, mel_freqs / 1000, mel_db, shading="auto", cmap=cmap)
    ax.set_ylabel("mel freq (kHz)", fontsize=8)
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.tick_params(labelsize=7)
    for side in ax.spines.values():
        side.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.008)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(cbar_label, fontsize=6.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial-dir", required=True)
    ap.add_argument("--mic", choices=["watch", "surface"], default="surface")
    ap.add_argument("--target-sr", type=int, default=16000)
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--hop-length", type=int, default=256)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--out", default="figure_audio_pipeline.pdf")
    ap.add_argument("--also-png", action="store_true")
    args = ap.parse_args()

    set_paper_style()
    wav_name = "watch_audio.wav" if args.mic == "watch" else "surface_mic.wav"
    wav_path = os.path.join(args.trial_dir, wav_name)
    events_path = os.path.join(args.trial_dir, "events.csv")
    color = WAVE_COLOR[args.mic]

    # (1) raw
    sr_raw, wave_raw = load_wav(wav_path)

    # (2) trim to [first touch-on, last touch-off]
    trim = get_trim_range(events_path) if os.path.exists(events_path) else None
    if trim is None:
        print("[WARN] No usable touch-on/off events found; skipping trim (using full trial).")
        wave_trim = wave_raw
        trim = (0, len(wave_raw) / sr_raw)
    else:
        on_t, off_t = trim
        i0, i1 = int(on_t * sr_raw), int(off_t * sr_raw)
        wave_trim = wave_raw[max(i0, 0):min(i1, len(wave_raw))]

    # (3) resample
    wave_rs = resample_audio(wave_trim, sr_raw, args.target_sr)

    # (4) log-mel spectrogram
    mel_db, t_spec, mel_freqs = log_mel_spectrogram(
        wave_rs, args.target_sr, args.n_fft, args.hop_length, args.n_mels)

    # (5) z-score normalize the spectrogram
    mel_z = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

    fig, axes = plt.subplots(
        5, 1, figsize=(7.2, 10.5),
        gridspec_kw={"height_ratios": [1, 1, 1, 1.6, 1.6]})
    ax1, ax2, ax3, ax4, ax5 = axes

    plot_waveform(ax1, wave_raw, sr_raw, color,
                  f"(1) Raw \u2014 {args.mic} mic, {sr_raw} Hz, {len(wave_raw)/sr_raw:.2f}s",
                  trim_lines=trim)
    plot_waveform(ax2, wave_trim, sr_raw, color,
                  f"(2) Trimmed to touch-on\u2192touch-off \u2014 {len(wave_trim)/sr_raw:.2f}s")
    plot_waveform(ax3, wave_rs, args.target_sr, color,
                  f"(3) Resampled to {args.target_sr} Hz \u2014 {len(wave_rs)/args.target_sr:.2f}s")
    plot_spectrogram(fig, ax4, mel_db, t_spec, mel_freqs,
                      f"(4) Log-mel spectrogram \u2014 n_fft={args.n_fft}, hop={args.hop_length}, "
                      f"n_mels={args.n_mels}")
    plot_spectrogram(fig, ax5, mel_z, t_spec, mel_freqs,
                      "(5) Z-score normalized", cbar_label="z-score")

    axes[-1].set_xlabel("time (s)", fontsize=9)
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    participant = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(args.trial_dir))))
    label = os.path.basename(os.path.dirname(args.trial_dir))
    trial = os.path.basename(args.trial_dir)
    fig.suptitle(f"Audio preprocessing pipeline \u2014 {participant} / '{label}' / {trial} "
                 f"({args.mic} mic)", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[DONE] Saved figure to {args.out}")
    if args.also_png:
        png_path = os.path.splitext(args.out)[0] + ".png"
        fig.savefig(png_path, dpi=400, bbox_inches="tight")
        print(f"[DONE] Saved high-res PNG to {png_path}")


if __name__ == "__main__":
    main()