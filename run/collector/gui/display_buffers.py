"""
wristpad/gui/display_buffers.py
────────────────────────────────────────────────────────────────────────────
Ring-buffer classes that feed the instructor window's live plots. These
never touch disk — the session CSVs/WAVs (see session.py / writers.py) are
the source of truth; instances of these classes are just a live view of the
same data, held in state.py.
"""

import threading
import time
from collections import deque

import numpy as np


class ScrollingWaveform:
    """Fixed-duration ring buffer for a 1-D audio/signal stream, with
    optional decimation on ingestion so the GUI isn't pushed thousands of
    points per line series at high sample rates (e.g. 192kHz surface mic)."""

    def __init__(self, sr: float, window_sec: float, decimate: int = 1):
        self.sr = sr
        self.decimate = max(1, decimate)
        self.dt = self.decimate / sr
        maxlen = max(2, int(window_sec * sr / self.decimate))
        self.buf = deque(maxlen=maxlen)

    def push(self, samples: np.ndarray):
        if self.decimate > 1:
            samples = samples[::self.decimate]
        self.buf.extend(samples.astype(np.float32).tolist())

    def get_xy(self):
        """Returns (x, y) with x in seconds *relative to the most recent
        sample* (0 = now, negative = in the past)."""
        n = len(self.buf)
        if n == 0:
            return np.array([0.0]), np.array([0.0])
        y = np.asarray(self.buf, dtype=np.float32)
        x = (np.arange(n) - (n - 1)) * self.dt
        return x, y


class ScrollingSpectrogram:
    """Rolling STFT image (freq bins x time columns), computed by hopping a
    Hann-windowed FFT over an incoming raw-sample stream."""

    def __init__(self, sr: float, n_fft: int = 1024, hop_sec: float = 0.02,
                 max_cols: int = 300, freq_max: float | None = None,
                 db_floor: float = -100.0):
        self.sr = sr
        self.n_fft = n_fft
        self.hop = max(1, int(hop_sec * sr))
        self.window = np.hanning(n_fft).astype(np.float32)
        self.raw_buf = deque(maxlen=n_fft)
        self._since_hop = 0
        self.db_floor = db_floor
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        freq_max = sr / 2 if freq_max is None else freq_max
        self.freq_mask = freqs <= freq_max
        self.freqs = freqs[self.freq_mask]
        self.n_freq = int(self.freq_mask.sum())
        self.max_cols = max_cols
        self.image = np.full((self.n_freq, max_cols), db_floor, dtype=np.float32)
        self._img_lock = threading.Lock()

    def push(self, samples: np.ndarray):
        self.raw_buf.extend(samples.astype(np.float32).tolist())
        self._since_hop += len(samples)
        while self._since_hop >= self.hop and len(self.raw_buf) >= self.n_fft // 2:
            self._since_hop -= self.hop
            arr = np.array(self.raw_buf, dtype=np.float32)
            if len(arr) < self.n_fft:
                arr = np.pad(arr, (self.n_fft - len(arr), 0))
            spec = np.abs(np.fft.rfft(arr * self.window))[self.freq_mask]
            db = np.clip(20.0 * np.log10(spec + 1e-6), self.db_floor, None)
            with self._img_lock:
                self.image = np.roll(self.image, -1, axis=1)
                self.image[:, -1] = db

    def get_image(self):
        with self._img_lock:
            return self.image.copy()


class ScrollingIMU:
    """Fixed-duration ring buffer for a 3-axis signal (acc or gyro)."""

    def __init__(self, window_sec: float, expected_hz: float = 100.0):
        maxlen = max(4, int(window_sec * expected_hz))
        self.t = deque(maxlen=maxlen); self.x = deque(maxlen=maxlen)
        self.y = deque(maxlen=maxlen); self.z = deque(maxlen=maxlen)
        self._t0 = None

    def push(self, v1, v2, v3, ts: float | None = None):
        """`ts`, if given, must be in the same time domain as
        time.perf_counter() — lets a caller that knows the true per-sample
        timing supply it instead of the moment push() happened to be
        called."""
        now = ts if ts is not None else time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        self.t.append(now - self._t0)
        self.x.append(v1); self.y.append(v2); self.z.append(v3)

    def get_series(self):
        if not self.t:
            return np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])
        t = np.asarray(self.t, dtype=np.float64)
        t = t - t[-1]
        return t, np.asarray(self.x), np.asarray(self.y), np.asarray(self.z)

    def get_rate_hz(self) -> float:
        """Average arrival rate over the current buffer — useful to tell
        "sensor genuinely quiet" from "packets stopped arriving"."""
        if len(self.t) < 2:
            return 0.0
        span = self.t[-1] - self.t[0]
        return (len(self.t) - 1) / span if span > 0 else 0.0
