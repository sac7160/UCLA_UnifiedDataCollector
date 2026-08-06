import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav


def load_and_normalize(filepath):
    """Loads a WAV file, converts it to mono if stereo, 
    and returns the sample rate, time axis in seconds, and normalized data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    sample_rate, data = wav.read(filepath)

    # Use only the first channel if the audio is stereo/multi-channel
    if len(data.shape) > 1:
        data = data[:, 0]

    # Normalize amplitude to the range [-1.0, 1.0]
    data = data.astype(np.float32)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    # Generate time axis in seconds
    duration = len(data) / sample_rate
    time_axis = np.linspace(0, duration, len(data))

    return sample_rate, time_axis, data


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and verify the synchronization between two WAV files."
    )
    parser.add_argument(
        "--a", required=True, help="Path to the first WAV file (e.g., surface_mic.wav)"
    )
    parser.add_argument(
        "--b", required=True, help="Path to the second WAV file (e.g., watch_audio.wav)"
    )
    args = parser.parse_args()

    # 1. Load and preprocess audio data
    try:
        sr_a, time_a, data_a = load_and_normalize(args.a)
        sr_b, time_b, data_b = load_and_normalize(args.b)
    except Exception as e:
        print(f"Error loading audio files: {e}")
        return

    print(f"[File A] {os.path.basename(args.a)} | SR: {sr_a}Hz | Duration: {time_a[-1]:.3f}s")
    print(f"[File B] {os.path.basename(args.b)} | SR: {sr_b}Hz | Duration: {time_b[-1]:.3f}s")

    # 2. Configure PyPlot visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("Audio Synchronization Verification", fontsize=16, fontweight="bold")

    # [Plot 1] File A Individual Waveform
    axes[0].plot(time_a, data_a, color="royalblue", alpha=0.8, label=os.path.basename(args.a))
    axes[0].set_ylabel("Normalized Amp")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Signal A (Reference)")

    # [Plot 2] File B Individual Waveform
    axes[1].plot(time_b, data_b, color="darkorange", alpha=0.8, label=os.path.basename(args.b))
    axes[1].set_ylabel("Normalized Amp")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")
    axes[1].set_title("Signal B (Target)")

    # [Plot 3] Overlay View (Crucial for verifying sync)
    axes[2].plot(time_a, data_a, color="royalblue", alpha=0.6, label=os.path.basename(args.a))
    axes[2].plot(time_b, data_b, color="darkorange", alpha=0.6, label=os.path.basename(args.b))
    axes[2].set_xlabel("Time (seconds)", fontsize=12)
    axes[2].set_ylabel("Normalized Amp")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right")
    axes[2].set_title("Overlay View (Zoom in here to check precise sync offset)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()