"""
realtime_multimodal_viz.py
────────────────────────────────────────────────────────────────────────────
WristPad touch-detection feasibility check — realtime multimodal monitor.

Streams shown (2x4 grid + tall camera column, PyQtGraph):
  col0: Surface mic waveform / Watch mic waveform / Watch IMU acc / Fingertip IMU acc
  col1: Surface mic spectrogram / Watch mic spectrogram / Watch IMU gyro / Fingertip IMU gyro
  col2 (tall): live camera preview with hand-skeleton + axis overlay

This is a *monitoring-only* tool — nothing is written to disk. Purpose is to
visually check whether contact-mic (surface mic) signal reacts cleanly to
finger-down/up events, as a precursor to replacing the spacebar marking in
unified_collector.py with automatic touch detection from that signal.

Reuses the watch TCP protocol, mic capture, and camera/fingertip-tracking
approach from unified_collector.py (persistent TCP connection with
length-prefixed frames; separate OS process for MediaPipe to avoid GIL
contention with the audio callback — same rationale as there).

Rendering: PyQtGraph instead of matplotlib. matplotlib's Agg backend is a
CPU software rasterizer redrawing full canvases every frame — fine for
publication figures, not built for a 9-panel real-time DAQ dashboard.
PyQtGraph is a Qt/(optionally OpenGL)-backed plotting library built
specifically for this (oscilloscope-style scientific instrumentation), so
this should feel meaningfully smoother without needing the blit
workarounds the matplotlib version needed.

Usage:
    python realtime_multimodal_viz.py
    python realtime_multimodal_viz.py --mic-device 1 --mic-channel 1
    python realtime_multimodal_viz.py --list-devices
    python realtime_multimodal_viz.py --no-camera                # skip fingertip IMU + preview
    python realtime_multimodal_viz.py --finger index --window-sec 4
    python realtime_multimodal_viz.py --opengl                   # try OpenGL-accelerated curves

Requires: pyqtgraph, PyQt5 (or PySide2/PyQt6/PySide6 — pyqtgraph auto-detects
whichever is installed), sounddevice, numpy, scipy, opencv-python, and
fingertip_imu_multi.py (from the WristPad repo) importable on PYTHONPATH.

    pip install pyqtgraph PyQt5
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import queue
import socket
import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt, sosfilt_zi

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
# NOTE: cv2 is imported lazily inside _camera_process_fn (the child process)
# only, so `--no-camera` runs don't require opencv-python to be installed.

# ─── PyQtGraph global config: white background, black foreground (paper-ish
# look), antialiasing off by default (antialiasing is one of the more
# expensive things a real-time line plot can do — off by default here,
# toggle with --antialias if you want prettier-but-slower lines) ───────────
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=False)
pg.setConfigOptions(imageAxisOrder='row-major')   # matplotlib-like (row=y, col=x) image arrays

_AXIS_COLORS = {'x': '#d62728', 'y': '#2ca02c', 'z': '#1f77b4'}

# ─── Config (mirrors unified_collector.py) ──────────────────────────────────
WATCH_HOST       = '0.0.0.0'
WATCH_PORT       = 50005
WATCH_AUDIO_SR   = 48000
WATCH_FRAME_SIZE = WATCH_AUDIO_SR // 25   # 1920 samples
WATCH_BUF_SIZE   = WATCH_FRAME_SIZE * 2   # 3840 bytes

MIC_SR         = 192000
MIC_CHANNELS   = 4
MIC_BLOCK_SIZE = 512

FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']

stop_event = threading.Event()


# ─── Small helpers: scrolling waveform + scrolling spectrogram buffers ──────
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
        self._skip_counter = 0

    def push(self, samples: np.ndarray):
        if self.decimate > 1:
            samples = samples[::self.decimate]
        self.buf.extend(samples.astype(np.float32).tolist())

    def get_xy(self):
        """Returns (x, y) with x in seconds *relative to the most recent
        sample* (0 = now, negative = in the past) — a standard scrolling
        oscilloscope convention, so the plot doesn't need its x-limits
        rescaled as the buffer fills up."""
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
        self._lock = threading.Lock()

    def push(self, samples: np.ndarray):
        self.raw_buf.extend(samples.astype(np.float32).tolist())
        self._since_hop += len(samples)
        while self._since_hop >= self.hop and len(self.raw_buf) >= self.n_fft // 2:
            self._since_hop -= self.hop
            arr = np.array(self.raw_buf, dtype=np.float32)
            if len(arr) < self.n_fft:
                arr = np.pad(arr, (self.n_fft - len(arr), 0))
            spec = np.abs(np.fft.rfft(arr * self.window))[self.freq_mask]
            db = 20.0 * np.log10(spec + 1e-6)
            db = np.clip(db, self.db_floor, None)
            with self._lock:
                self.image = np.roll(self.image, -1, axis=1)
                self.image[:, -1] = db

    def get_image(self):
        """Returns a (n_freq x max_cols) dB image, row 0 = lowest freq, ready
        to hand straight to imshow(..., origin='lower')."""
        with self._lock:
            return self.image.copy()

    def extent(self, window_sec: float):
        """(xmin, xmax, ymin, ymax) for imshow, in (seconds-relative-to-now, Hz)."""
        return (-window_sec, 0.0, 0.0, float(self.freqs[-1]))


class ScrollingIMU:
    """Fixed-duration ring buffer for a 3-axis signal (acc or gyro)."""

    def __init__(self, window_sec: float, expected_hz: float = 100.0):
        maxlen = max(4, int(window_sec * expected_hz))
        self.t  = deque(maxlen=maxlen)
        self.x  = deque(maxlen=maxlen)
        self.y  = deque(maxlen=maxlen)
        self.z  = deque(maxlen=maxlen)
        self._t0 = None

    def push(self, v1, v2, v3, ts: float | None = None):
        """`ts`, if given, must be in the same time domain as
        time.perf_counter() — lets a caller that knows the true per-sample
        timing (e.g. reconstructing spacing within a batched network packet)
        supply it instead of the moment push() happened to be called."""
        now = ts if ts is not None else time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        self.t.append(now - self._t0)
        self.x.append(v1); self.y.append(v2); self.z.append(v3)

    def get_series(self):
        """Same relative-to-now convention as ScrollingWaveform.get_xy()."""
        if not self.t:
            return np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])
        t = np.asarray(self.t, dtype=np.float64)
        t = t - t[-1]
        return t, np.asarray(self.x), np.asarray(self.y), np.asarray(self.z)

    def get_rate_hz(self) -> float:
        """Average arrival rate over the current buffer — a flat-looking
        segment with a healthy Hz here means the sensor itself was reading
        near-constant (e.g. a still wrist), not that packets stopped
        arriving. A rate near 0 means the opposite: a transmission stall."""
        if len(self.t) < 2:
            return 0.0
        span = self.t[-1] - self.t[0]
        return (len(self.t) - 1) / span if span > 0 else 0.0

    def seconds_since_last_sample(self) -> float | None:
        """How long ago the most recent sample arrived — catches an
        in-progress stall even before it's dragged the windowed rate down."""
        if not self.t or self._t0 is None:
            return None
        return (time.perf_counter() - self._t0) - self.t[-1]


# ─── Global buffers ──────────────────────────────────────────────────────────
_buf_lock = threading.Lock()

surface_wave: ScrollingWaveform | None = None
surface_spec: ScrollingSpectrogram | None = None
watch_wave:   ScrollingWaveform | None = None
watch_spec:   ScrollingSpectrogram | None = None

watch_acc = ScrollingIMU(window_sec=5.0, expected_hz=100.0)
watch_gyro = ScrollingIMU(window_sec=5.0, expected_hz=100.0)
finger_acc = ScrollingIMU(window_sec=5.0, expected_hz=30.0)
finger_gyro = ScrollingIMU(window_sec=5.0, expected_hz=30.0)

_target_finger = 'index'   # updated from CLI / GUI combo

_watch_rms = 0.0
_mic_rms = 0.0

# ── touch-detection signal: band-limited envelope relative to an adaptively
# tracked noise floor, in dB. This drives the touch ON/OFF indicator instead
# of raw broadband RMS — see the long comment above _mic_callback for why. ──
TOUCH_BAND_LOW_HZ  = 300.0     # matches the passive-acoustic touch band from earlier feasibility work
TOUCH_BAND_HIGH_HZ = 3000.0
ENV_ATTACK_TAU_SEC  = 0.005    # envelope rises fast on touch onset
ENV_RELEASE_TAU_SEC = 0.08     # ...and falls more slowly, so brief gaps during a drag don't drop out
FLOOR_RISE_TAU_SEC  = 2.0      # noise floor estimate creeps upward slowly, but drops instantly

_mic_band_sos = None    # built in main() once the actual mic sample rate is known
_mic_band_zi  = None
_mic_sr_runtime = MIC_SR
_envelope = 0.0
_floor_envelope = 0.0   # envelope tracked from the RAW (non-median-filtered) signal — see _mic_callback
_noise_floor = None
_touch_metric_db = -60.0   # envelope level relative to the noise floor, in dB

# ── outlier rejection + debounce, computed here (audio thread) at precise
# block timing rather than at GUI frame rate, so the decision doesn't depend
# on --fps and isn't limited to whatever the GUI happened to poll. ──────────
_touch_median_buf: "deque | None" = None   # built in main() once --touch-median-window is known
_touch_on_threshold_db  = 6.0   # mirrored from the GUI spinboxes; also the CLI --touch-threshold default
_touch_off_threshold_db = 3.0   # mirrored from the GUI spinboxes
_touch_min_on_sec  = 0.03       # metric must stay >= on-threshold this long before flipping ON
_touch_min_off_sec = 0.1       # metric must stay <  off-threshold this long before flipping OFF
_touch_on_state = False
_touch_candidate_on_time  = 0.0
_touch_candidate_off_time = 0.0


# ─── Surface mic (sounddevice) ──────────────────────────────────────────────
_mic_sos = butter(2, 10.0 / (MIC_SR / 2), btype='high', output='sos')
_mic_zi = sosfilt_zi(_mic_sos) * 0
_mic_target_ch = 0   # 0-indexed, set from args


def _mic_callback(indata, frames, time_info, status):
    """Two parallel paths per block:
      - the existing DC-blocked (10Hz highpass) signal, for the waveform/
        spectrogram display and the legacy _mic_rms readout.
      - a touch-detection path: band-pass to the ~300-3000Hz range actual
        finger contact energy lives in (cutting out low-frequency handling/
        room noise that raw broadband RMS was picking up); a short median
        filter to reject single-block outlier spikes (a knock, a click —
        anything that doesn't persist); an attack/release envelope follower
        on top of that (bridges brief dips during a drag); expressed
        *relative to* a slowly-adapting noise floor rather than as an
        absolute level (so the threshold means roughly the same thing
        regardless of mic gain/room noise/how hard a given touch is); and
        finally a debounce state machine that only flips ON/OFF once the
        metric has stayed past the threshold for a minimum duration — this
        is what actually protects against jitter/false positives, more than
        any single amount of smoothing upstream of it does.
    """
    global _mic_zi, _mic_rms, _mic_band_zi, _envelope, _floor_envelope, _noise_floor, _touch_metric_db
    global _touch_on_state, _touch_candidate_on_time, _touch_candidate_off_time
    if status:
        print(f'\n[MIC] {status}')
    raw = indata[:, _mic_target_ch].astype(np.float32)

    filtered, _mic_zi = sosfilt(_mic_sos, raw, zi=_mic_zi)
    amplified = np.clip(filtered, -1.0, 1.0)
    _mic_rms = float(np.sqrt(np.mean(amplified ** 2)))
    surface_wave.push(amplified)
    surface_spec.push(amplified)

    if _mic_band_sos is not None:
        band, _mic_band_zi = sosfilt(_mic_band_sos, raw, zi=_mic_band_zi)
        block_energy_raw = float(np.sqrt(np.mean(band ** 2)))
        block_dt = frames / _mic_sr_runtime

        # median-of-last-N block energies — rejects single-block outliers
        # (e.g. an incidental knock) that a linear average would still let
        # through partially. Used for the *decision* envelope only.
        block_energy = block_energy_raw
        if _touch_median_buf is not None:
            _touch_median_buf.append(block_energy_raw)
            block_energy = float(np.median(_touch_median_buf))

        coef_attack  = float(np.exp(-block_dt / ENV_ATTACK_TAU_SEC))
        coef_release = float(np.exp(-block_dt / ENV_RELEASE_TAU_SEC))
        
        # Decision-facing envelope: built from the median-filtered value,
        # so a single-block spike (a knock) can't trigger a false ON.
        if block_energy > _envelope:
            _envelope = coef_attack * _envelope + (1.0 - coef_attack) * block_energy
        else:
            _envelope = coef_release * _envelope + (1.0 - coef_release) * block_energy

        # Floor-facing envelope: built from raw signal energy
        if block_energy_raw > _floor_envelope:
            _floor_envelope = coef_attack * _floor_envelope + (1.0 - coef_attack) * block_energy_raw
        else:
            _floor_envelope = coef_release * _floor_envelope + (1.0 - coef_release) * block_energy_raw

        # ---------------------------------------------------------------------
        # [수정] 노이즈 플로어 고정 (Fixed Noise Floor)
        # ---------------------------------------------------------------------
        if _noise_floor is None:
            _noise_floor = _floor_envelope
        elif _floor_envelope < _noise_floor:
            # 프로그램 시작 직후나 기존보다 더 조용한 상태(진짜 바닥)가 발견될 때만 기준점 하향
            # 한 번 방의 최저 소음(Absolute Minimum)을 찾으면 더 이상 올라가지 않고 고정됨
            _noise_floor = 0.8 * _floor_envelope + 0.2 * _noise_floor

        _touch_metric_db = 20.0 * np.log10((_envelope + 1e-8) / (_noise_floor + 1e-8))

        # Debounce state machine: commit state flip once sustained past thresholds
        now_pc = time.perf_counter()
        if not _touch_on_state:
            if _touch_metric_db >= _touch_on_threshold_db:
                _touch_candidate_on_time += block_dt
                if _touch_candidate_on_time >= _touch_min_on_sec:
                    _touch_on_state = True
                    true_on_pc = now_pc - _touch_candidate_on_time
                    print(f'[TOUCH] ON  — true onset ~{true_on_pc:.3f}s (perf_counter), '
                          f'confirmed after {_touch_candidate_on_time*1000:.0f}ms')
                    _touch_candidate_on_time = 0.0
            else:
                _touch_candidate_on_time = 0.0
        else:
            if _touch_metric_db < _touch_off_threshold_db:
                _touch_candidate_off_time += block_dt
                if _touch_candidate_off_time >= _touch_min_off_sec:
                    _touch_on_state = False
                    true_off_pc = now_pc - _touch_candidate_off_time
                    print(f'[TOUCH] OFF — true offset ~{true_off_pc:.3f}s (perf_counter), '
                          f'confirmed after {_touch_candidate_off_time*1000:.0f}ms')
                    _touch_candidate_off_time = 0.0
            else:
                _touch_candidate_off_time = 0.0


# ─── Watch TCP listener (adapted from unified_collector.py, no file I/O) ────
def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        if stop_event.is_set():
            return None
        try:
            chunk = conn.recv(n - len(buf))
        except socket.timeout:
            continue
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _parse_imu_packet(pkt: bytes, sensor: str):
    try:
        txt = pkt[5:].decode('utf-8', errors='ignore').strip()
    except Exception:
        return

    samples = []
    for sample in txt.split('|'):
        parts = sample.strip().split()
        if len(parts) < 3:
            continue
        try:
            v1, v2, v3 = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        watch_ts_ms = float(parts[3]) if len(parts) >= 4 else None
        samples.append((v1, v2, v3, watch_ts_ms))
    if not samples:
        return

    target = watch_acc if sensor == 'acc' else watch_gyro
    arrival_pc = time.perf_counter()

    if len(samples) > 1 and all(s[3] is not None for s in samples):
        last_watch_sec = samples[-1][3] / 1000.0
        for v1, v2, v3, watch_ts_ms in samples:
            pc_ts = arrival_pc - (last_watch_sec - watch_ts_ms / 1000.0)
            target.push(v1, v2, v3, ts=pc_ts)
    else:
        for v1, v2, v3, _ in samples:
            target.push(v1, v2, v3, ts=arrival_pc)


def _dispatch_watch_packet(pkt: bytes):
    global _watch_rms
    total = len(pkt)

    if total == WATCH_BUF_SIZE + 8:
        audio_bytes = pkt[8:]
    elif total == WATCH_BUF_SIZE:
        audio_bytes = pkt
    else:
        audio_bytes = None

    if audio_bytes is not None:
        samples = np.frombuffer(audio_bytes, dtype='<i2').astype(np.float32) / 32768.0
        _watch_rms = float(np.sqrt(np.mean(samples ** 2)))
        watch_wave.push(samples)
        watch_spec.push(samples)
        return

    try:
        hdr = pkt[:5].decode('utf-8', errors='ignore')
    except Exception:
        return

    if hdr == 'IMUAC':
        _parse_imu_packet(pkt, 'acc')
    elif hdr == 'IMUGY':
        _parse_imu_packet(pkt, 'gyro')
    elif hdr in ('SUBID', 'RTBGN', 'RTEND'):
        pass
    elif total > 0 and total % 2 == 0:
        samples = np.frombuffer(pkt, dtype='<i2').astype(np.float32) / 32768.0
        watch_wave.push(samples)
        watch_spec.push(samples)


def _net_thread_fn(watch_port: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((WATCH_HOST, watch_port))
    srv.listen(16)
    srv.settimeout(1.0)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = 'unknown'
    print(f'[NET] Watch TCP: {local_ip}:{watch_port}')

    while not stop_event.is_set():
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            continue
        except Exception as e:
            if not stop_event.is_set():
                print(f'[NET] accept error: {e}')
            continue

        print(f'[NET] watch connected from {addr}')
        conn.settimeout(1.0)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while not stop_event.is_set():
                header = _recv_exact(conn, 4)
                if header is None:
                    break
                msg_len = int.from_bytes(header, byteorder='big', signed=False)
                if msg_len <= 0 or msg_len > 10_000_000:
                    print(f'[NET] implausible message length {msg_len}, dropping connection')
                    break
                payload = _recv_exact(conn, msg_len)
                if payload is None:
                    break
                _dispatch_watch_packet(payload)
        except Exception as e:
            print(f'[NET] connection error: {e}')
        finally:
            conn.close()
            print('[NET] watch disconnected — waiting for reconnect')

    srv.close()
    print('[NET] stopped')


# ─── Camera process (fingertip virtual IMU + preview frame) ─────────────────
def _camera_process_fn(camera_index: int, camera_pitch_deg, camera_roll_deg: float,
                        record_queue: "mp.Queue", frame_queue: "mp.Queue",
                        stop_flag: "mp.Event"):
    import cv2 as _cv2
    from fingertip_imu_multi import MultiFingertipIMUTracker, gravity_vector_from_camera_tilt

    gravity_mm_s2 = None
    if camera_pitch_deg is not None:
        gravity_mm_s2 = gravity_vector_from_camera_tilt(camera_pitch_deg, camera_roll_deg)

    tracker = MultiFingertipIMUTracker(max_num_hands=1, gravity_mm_s2=gravity_mm_s2)
    cap = _cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f'[CAM] failed to open camera index={camera_index}')
        record_queue.put(None)
        return
    print(f'[CAM] camera index={camera_index} opened OK '
          f'({int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))})')

    t0 = time.perf_counter()
    preview_frame_i = 0
    while not stop_flag.is_set():
        success, frame = cap.read()
        if not success:
            continue
        frame = _cv2.flip(frame, 1)
        ts = time.perf_counter() - t0
        records = tracker.update(frame, timestamp=ts)

        try:
            record_queue.put_nowait(records)
        except Exception:
            pass

        tracker.draw(frame)
        tracker.draw_axes(frame)

        preview_frame_i += 1
        if preview_frame_i % 2 == 0:
            small = _cv2.resize(frame, (0, 0), fx=0.6, fy=0.6, interpolation=_cv2.INTER_AREA)
            rgb = _cv2.cvtColor(small, _cv2.COLOR_BGR2RGB)
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Exception:
                    pass
            try:
                frame_queue.put_nowait(rgb)
            except Exception:
                pass

    cap.release()
    tracker.close()


def _camera_bridge_thread_fn(record_queue: "mp.Queue"):
    while not stop_event.is_set():
        try:
            records = record_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if records is None:
            print('[CAM] camera process reported it could not open the camera')
            continue
        for r in records:
            if r.finger != _target_finger:
                continue
            if not r.detected:
                continue
            acc_vals = (r.accel_x, r.accel_y, r.accel_z)
            gyro_vals = (r.gyro_x, r.gyro_y, r.gyro_z)
            if all(np.isfinite(v) for v in acc_vals):
                finger_acc.push(*acc_vals)
            if all(np.isfinite(v) for v in gyro_vals):
                finger_gyro.push(*gyro_vals)


# ─── PyQtGraph dashboard ─────────────────────────────────────────────────────
class LiveDashboard(QtWidgets.QMainWindow):
    def __init__(self, window_sec: float, frame_queue, has_camera: bool,
                 use_opengl: bool = False, touch_threshold_db: float = 6.0,
                 touch_hysteresis_db: float = 3.0):
        super().__init__()
        self.window_sec = window_sec
        self.frame_queue = frame_queue
        self.has_camera = has_camera
        self._last_frame_shape = None
        self._metric_min = None
        self._metric_max = None

        self.setWindowTitle('WristPad Realtime Multimodal Monitor')
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        # Top control bar
        top_bar = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel('')
        self.status_label.setStyleSheet('font-size: 12px; color: #333;')
        top_bar.addWidget(self.status_label)
        top_bar.addStretch(1)
        top_bar.addWidget(QtWidgets.QLabel('touch threshold (dB above noise floor):'))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setDecimals(1)
        self.threshold_spin.setRange(-10.0, 60.0)
        self.threshold_spin.setSingleStep(0.5)
        self.threshold_spin.setValue(touch_threshold_db)
        top_bar.addWidget(self.threshold_spin)

        top_bar.addWidget(QtWidgets.QLabel('hysteresis (dB):'))
        self.hysteresis_spin = QtWidgets.QDoubleSpinBox()
        self.hysteresis_spin.setDecimals(1)
        self.hysteresis_spin.setRange(0.0, 30.0)
        self.hysteresis_spin.setSingleStep(0.5)
        self.hysteresis_spin.setValue(touch_hysteresis_db)
        top_bar.addWidget(self.hysteresis_spin)
        outer.addLayout(top_bar)

        self.threshold_spin.valueChanged.connect(self._sync_touch_thresholds)
        self.hysteresis_spin.valueChanged.connect(self._sync_touch_thresholds)
        self._sync_touch_thresholds()

        grid = QtWidgets.QGridLayout()
        outer.addLayout(grid)

        self.pw_surface_wave = self._make_waveform_plot('Surface mic — waveform')
        self.pw_watch_wave   = self._make_waveform_plot('Watch mic — waveform')
        self.pw_surface_spec, self.img_surface_spec = self._make_spec_plot(
            'Surface mic — spectrogram', surface_spec)
        self.pw_watch_spec, self.img_watch_spec = self._make_spec_plot(
            'Watch mic — spectrogram', watch_spec)
        self.pw_wacc,  self.curves_wacc  = self._make_imu_plot('Watch IMU — acc')
        self.pw_wgyro, self.curves_wgyro = self._make_imu_plot('Watch IMU — gyro')
        self.pw_facc,  self.curves_facc  = self._make_imu_plot(f'Fingertip IMU ({_target_finger}) — acc')
        self.pw_fgyro, self.curves_fgyro = self._make_imu_plot(f'Fingertip IMU ({_target_finger}) — gyro')

        if use_opengl:
            for pw in (self.pw_surface_wave, self.pw_watch_wave, self.pw_wacc,
                       self.pw_wgyro, self.pw_facc, self.pw_fgyro):
                try:
                    pw.useOpenGL(True)
                except Exception:
                    pass

        grid.addWidget(self.pw_surface_wave, 0, 0)
        grid.addWidget(self.pw_surface_spec, 0, 1)
        grid.addWidget(self.pw_watch_wave,   1, 0)
        grid.addWidget(self.pw_watch_spec,   1, 1)
        grid.addWidget(self.pw_wacc,  2, 0)
        grid.addWidget(self.pw_wgyro, 2, 1)
        grid.addWidget(self.pw_facc,  3, 0)
        grid.addWidget(self.pw_fgyro, 3, 1)

        # Right column: camera preview & touch status
        self.cam_label = QtWidgets.QLabel('waiting for camera…' if has_camera else '--no-camera specified')
        self.cam_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cam_label.setStyleSheet('background-color: black; color: white; font-size: 12px;')
        self.cam_label.setMinimumWidth(420)

        self.touch_label = QtWidgets.QLabel()
        self.touch_label.setAlignment(QtCore.Qt.AlignCenter)
        font = self.touch_label.font()
        font.setPointSize(28)
        font.setBold(True)
        self.touch_label.setFont(font)
        self._set_touch_visual(False, metric_db=-60.0)

        right_col = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.cam_label, 1)
        right_layout.addWidget(self.touch_label, 1)

        minmax_row = QtWidgets.QHBoxLayout()
        self.minmax_label = QtWidgets.QLabel('since reset — min=–  max=–')
        self.minmax_label.setStyleSheet(
            'font-size: 14px; font-weight: bold; color: #222; '
            'background-color: #eee; padding: 4px; border-radius: 3px;')
        self.minmax_label.setAlignment(QtCore.Qt.AlignCenter)
        self.reset_minmax_btn = QtWidgets.QPushButton('reset min/max')
        self.reset_minmax_btn.clicked.connect(self._reset_rms_minmax)
        minmax_row.addWidget(self.minmax_label, 1)
        minmax_row.addWidget(self.reset_minmax_btn)
        right_layout.addLayout(minmax_row)

        grid.addWidget(right_col, 0, 2, 4, 1)

        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        self.resize(1600, 1000)

    def _make_waveform_plot(self, title: str) -> pg.PlotWidget:
        pw = pg.PlotWidget(title=title)
        pw.setLabel('bottom', 'time relative to now (s)')
        pw.setLabel('left', 'amplitude')
        pw.setXRange(-self.window_sec, 0, padding=0)
        pw.setYRange(-1.05, 1.05, padding=0)
        pw.showGrid(x=True, y=True, alpha=0.25)
        curve = pw.plot(pen=pg.mkPen('#333333', width=1))
        curve.setDownsampling(auto=True, method='peak')
        curve.setClipToView(True)
        pw._curve = curve
        return pw

    def _make_spec_plot(self, title: str, spec: "ScrollingSpectrogram"):
        pw = pg.PlotWidget(title=title)
        pw.setLabel('bottom', 'time relative to now (s)')
        pw.setLabel('left', 'frequency (Hz)')
        img = pg.ImageItem()
        try:
            cmap = pg.colormap.get('magma', source='matplotlib')
            img.setLookupTable(cmap.getLookupTable())
        except Exception:
            pass
        freq_max = float(spec.freqs[-1])
        img.setImage(spec.get_image(), autoLevels=True)
        img.setRect(QtCore.QRectF(-self.window_sec, 0, self.window_sec, freq_max))
        pw.addItem(img)
        pw.setXRange(-self.window_sec, 0, padding=0)
        pw.setYRange(0, freq_max, padding=0)
        return pw, img

    def _make_imu_plot(self, title: str, window_sec: float = 5.0):
        pw = pg.PlotWidget(title=title)
        pw.setLabel('bottom', 'time relative to now (s)')
        pw.setLabel('left', 'value')
        pw.setXRange(-window_sec, 0, padding=0)
        pw.showGrid(x=True, y=True, alpha=0.25)
        pw.addLegend(offset=(5, 5))
        curves = {}
        for axis_name in ('x', 'y', 'z'):
            c = pw.plot(pen=pg.mkPen(_AXIS_COLORS[axis_name], width=1), name=axis_name)
            c.setDownsampling(auto=True, method='peak')
            c.setClipToView(True)
            curves[axis_name] = c
        return pw, curves

    def _update_waveform(self, pw, waveform: "ScrollingWaveform"):
        x, y = waveform.get_xy()
        pw._curve.setData(x, y)
        if len(y) > 1:
            m = float(np.max(np.abs(y))) * 1.2
            pw.setYRange(-max(m, 0.02), max(m, 0.02), padding=0)

    def _update_spec(self, img, spec: "ScrollingSpectrogram"):
        img.setImage(spec.get_image(), autoLevels=True)

    def _update_imu(self, pw, curves, imu: "ScrollingIMU"):
        t, x, y, z = imu.get_series()
        curves['x'].setData(t, x)
        curves['y'].setData(t, y)
        curves['z'].setData(t, z)
        if len(t) <= 1:
            return
        allv = np.concatenate([x, y, z])
        finite = allv[np.isfinite(allv)]
        if finite.size == 0:
            return
        lo, hi = float(np.min(finite)), float(np.max(finite))
        pad = max((hi - lo) * 0.15, 1e-3)
        pw.setYRange(lo - pad, hi + pad, padding=0)

    def _update_camera(self):
        if self.frame_queue is None:
            return
        frame = None
        try:
            while True:
                frame = self.frame_queue.get_nowait()
        except queue.Empty:
            pass
        if frame is None:
            return
        h, w = frame.shape[:2]
        qimg = QtGui.QImage(frame.data, w, h, frame.strides[0], QtGui.QImage.Format_RGB888).copy()
        pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
            self.cam_label.width(), self.cam_label.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.cam_label.setPixmap(pixmap)

    def _set_touch_visual(self, is_on: bool, metric_db: float):
        if is_on:
            self.touch_label.setStyleSheet('background-color: #2ca02c; color: white;')
            self.touch_label.setText(f'TOUCH ON\n({metric_db:.1f} dB above floor)')
        else:
            self.touch_label.setStyleSheet('background-color: #d62728; color: white;')
            self.touch_label.setText(f'TOUCH OFF\n({metric_db:.1f} dB above floor)')

    def _sync_touch_thresholds(self):
        global _touch_on_threshold_db, _touch_off_threshold_db
        on = self.threshold_spin.value()
        hyst = self.hysteresis_spin.value()
        off = on - hyst
        _touch_on_threshold_db = on
        _touch_off_threshold_db = off
        if off <= 0.3:
            print(f'[TOUCH] WARNING: off-threshold={off:.1f}dB is at/below the floor\'s own '
                  f'baseline jitter — OFF may never confirm again once ON triggers. '
                  f'Lower the hysteresis instead of raising it, so off-threshold stays '
                  f'meaningfully positive (>= ~0.5-1.0dB).')

    def _reset_rms_minmax(self):
        self._metric_min = None
        self._metric_max = None

    def update(self):
        self._update_waveform(self.pw_surface_wave, surface_wave)
        self._update_waveform(self.pw_watch_wave, watch_wave)
        self._update_spec(self.img_surface_spec, surface_spec)
        self._update_spec(self.img_watch_spec, watch_spec)
        self._update_imu(self.pw_wacc, self.curves_wacc, watch_acc)
        self._update_imu(self.pw_wgyro, self.curves_wgyro, watch_gyro)
        self._update_imu(self.pw_facc, self.curves_facc, finger_acc)
        self._update_imu(self.pw_fgyro, self.curves_fgyro, finger_gyro)
        self._update_camera()
        self._set_touch_visual(_touch_on_state, _touch_metric_db)

        if np.isfinite(_touch_metric_db):
            self._metric_min = _touch_metric_db if self._metric_min is None else min(self._metric_min, _touch_metric_db)
            self._metric_max = _touch_metric_db if self._metric_max is None else max(self._metric_max, _touch_metric_db)
        if self._metric_min is not None:
            self.minmax_label.setText(
                f'since reset — min={self._metric_min:.1f}dB  max={self._metric_max:.1f}dB')
        else:
            self.minmax_label.setText('since reset — min=–  max=–')

        self.pw_wacc.setTitle(f'Watch IMU — acc ({watch_acc.get_rate_hz():.0f} Hz)')
        self.pw_wgyro.setTitle(f'Watch IMU — gyro ({watch_gyro.get_rate_hz():.0f} Hz)')
        self.pw_facc.setTitle(f'Fingertip IMU ({_target_finger}) — acc ({finger_acc.get_rate_hz():.0f} Hz)')
        self.pw_fgyro.setTitle(f'Fingertip IMU ({_target_finger}) — gyro ({finger_gyro.get_rate_hz():.0f} Hz)')
        self.status_label.setText(
            f'surface mic RMS={_mic_rms:.4f}    touch metric={_touch_metric_db:.1f}dB    '
            f'watch mic RMS={_watch_rms:.4f}    target finger={_target_finger}')

    def closeEvent(self, event):
        stop_event.set()
        super().closeEvent(event)


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    global surface_wave, surface_spec, watch_wave, watch_spec, _mic_target_ch, _target_finger

    parser = argparse.ArgumentParser(description='WristPad realtime multimodal monitor')
    parser.add_argument('--mic-device', type=int, default=None)
    parser.add_argument('--mic-channel', type=int, default=1)
    parser.add_argument('--mic-sr', type=int, default=MIC_SR)
    parser.add_argument('--watch-port', type=int, default=WATCH_PORT)
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--camera-pitch-deg', type=float, default=None)
    parser.add_argument('--camera-roll-deg', type=float, default=0.0)
    parser.add_argument('--no-camera', action='store_true')
    parser.add_argument('--finger', choices=FINGER_NAMES, default='index')
    parser.add_argument('--window-sec', type=float, default=2.0,
                         help='waveform/spectrogram display window (sec)')
    parser.add_argument('--fps', type=float, default=30.0,
                         help='GUI refresh rate (default 30).')
    parser.add_argument('--display-hz', type=int, default=8000,
                         help='waveform lines decimate down to this sample rate for display.')
    parser.add_argument('--opengl', action='store_true',
                         help='try OpenGL-accelerated line rendering.')
    parser.add_argument('--touch-threshold', type=float, default=6.0,
                         help='touch metric threshold above which = touch ON (default 6.0 dB).')
    parser.add_argument('--touch-hysteresis', type=float, default=3.0,
                         help='OFF-threshold = ON-threshold minus this many dB (default 3.0).')
    parser.add_argument('--touch-min-on-ms', type=float, default=30.0,
                         help='metric must stay above ON threshold continuously for this ms before ON.')
    parser.add_argument('--touch-min-off-ms', type=float, default=60.0,
                         help='metric must stay below OFF threshold continuously for this ms before OFF.')
    parser.add_argument('--touch-median-window', type=int, default=3,
                         help='median filter window in mic blocks applied before smoothing.')
    parser.add_argument('--touch-band-low', type=float, default=TOUCH_BAND_LOW_HZ,
                         help=f'lower edge of touch band-pass filter (default {TOUCH_BAND_LOW_HZ:.0f}Hz).')
    parser.add_argument('--touch-band-high', type=float, default=TOUCH_BAND_HIGH_HZ,
                         help=f'upper edge of touch band-pass filter (default {TOUCH_BAND_HIGH_HZ:.0f}Hz).')
    parser.add_argument('--list-devices', action='store_true')
    args = parser.parse_args()

    if args.list_devices:
        print('\n=== Audio Devices ===')
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                print(f'  [{i:2d}] {d["name"]} (in={d["max_input_channels"]}, sr={int(d["default_samplerate"])})')
        return

    _target_finger = args.finger

    mic_device = args.mic_device
    global_mic_target_ch = args.mic_channel
    if mic_device is None:
        for i, d in enumerate(sd.query_devices()):
            name = d['name'].lower()
            if ('tascam' in name or 'us-4x4' in name) and d['max_input_channels'] > 0:
                mic_device = i
                break
            if ('focusrite' in name or 'scarlett' in name) and d['max_input_channels'] > 0:
                mic_device = i
                if global_mic_target_ch == 1:
                    global_mic_target_ch = 2
                    print('[MIC] Scarlett detected → auto-set to channel 2 (XLR input)')
                break
    if mic_device is None:
        print('[MIC] No audio interface found. Specify one with --mic-device N (see --list-devices).')
        sys.exit(1)
    _mic_target_ch = global_mic_target_ch - 1
    print(f'[MIC] device=[{mic_device}] ch={global_mic_target_ch} sr={args.mic_sr}')

    surface_decimate = max(1, args.mic_sr // args.display_hz)
    surface_wave = ScrollingWaveform(args.mic_sr, args.window_sec, decimate=surface_decimate)
    surface_spec = ScrollingSpectrogram(args.mic_sr, n_fft=1024, hop_sec=0.02,
                                         max_cols=int(args.window_sec / 0.02), freq_max=10000.0)

    watch_decimate = max(1, WATCH_AUDIO_SR // args.display_hz)
    watch_wave = ScrollingWaveform(WATCH_AUDIO_SR, args.window_sec, decimate=watch_decimate)
    watch_spec = ScrollingSpectrogram(WATCH_AUDIO_SR, n_fft=1024, hop_sec=0.03,
                                       max_cols=int(args.window_sec / 0.03), freq_max=6000.0)

    global _mic_band_sos, _mic_band_zi, _mic_sr_runtime
    global _touch_median_buf, _touch_on_threshold_db, _touch_off_threshold_db
    global _touch_min_on_sec, _touch_min_off_sec
    _mic_sr_runtime = args.mic_sr
    nyquist = args.mic_sr / 2.0
    band_low = args.touch_band_low
    band_high = min(args.touch_band_high, nyquist - 100.0)
    if band_high > band_low:
        _mic_band_sos = butter(4, [band_low / nyquist, band_high / nyquist],
                                btype='band', output='sos')
        _mic_band_zi = sosfilt_zi(_mic_band_sos) * 0
        print(f'[TOUCH] band-pass filter: {band_low:.0f}-{band_high:.0f}Hz')
    else:
        print(f'[MIC] mic-sr={args.mic_sr} too low for the {band_low:.0f}-'
              f'{args.touch_band_high:.0f}Hz touch band — touch detection will stay at -inf dB.')

    _touch_median_buf = deque(maxlen=max(1, args.touch_median_window))
    _touch_on_threshold_db = args.touch_threshold
    _touch_off_threshold_db = args.touch_threshold - args.touch_hysteresis
    _touch_min_on_sec = args.touch_min_on_ms / 1000.0
    _touch_min_off_sec = args.touch_min_off_ms / 1000.0
    if _touch_off_threshold_db <= 0.3:
        print(f'[TOUCH] WARNING: off-threshold={_touch_off_threshold_db:.1f}dB is at/below the '
              f'floor\'s own baseline jitter — OFF may never confirm again once ON triggers. '
              f'Try a smaller --touch-hysteresis so off-threshold stays >= ~0.5-1.0dB.')

    mic_stream = sd.InputStream(
        device=mic_device, channels=MIC_CHANNELS, samplerate=args.mic_sr,
        blocksize=MIC_BLOCK_SIZE, dtype='float32', callback=_mic_callback,
    )
    mic_stream.start()

    net_t = threading.Thread(target=_net_thread_fn, args=(args.watch_port,), daemon=True)
    net_t.start()

    cam_proc = cam_bridge_t = record_queue = frame_queue = cam_stop_flag = None
    if not args.no_camera:
        record_queue = mp.Queue(maxsize=8)
        frame_queue  = mp.Queue(maxsize=1)
        cam_stop_flag = mp.Event()
        cam_proc = mp.Process(
            target=_camera_process_fn,
            args=(args.camera_index, args.camera_pitch_deg, args.camera_roll_deg,
                  record_queue, frame_queue, cam_stop_flag),
            daemon=True,
        )
        cam_proc.start()
        cam_bridge_t = threading.Thread(target=_camera_bridge_thread_fn, args=(record_queue,), daemon=True)
        cam_bridge_t.start()
        print(f'[CAM] started in a separate process (index={args.camera_index}, pid={cam_proc.pid})')
    else:
        print('[CAM] --no-camera specified → fingertip IMU panel and camera preview will stay empty')

    app = pg.mkQApp('WristPad Realtime Monitor')
    window = LiveDashboard(args.window_sec, frame_queue, has_camera=not args.no_camera,
                            use_opengl=args.opengl, touch_threshold_db=args.touch_threshold,
                            touch_hysteresis_db=args.touch_hysteresis)
    window.show()

    timer = QtCore.QTimer()
    timer.timeout.connect(window.update)
    timer.start(max(1, int(1000 / args.fps)))

    print('[RUN] Monitoring... close the window or Ctrl+C to stop')

    def _shutdown():
        stop_event.set()
        if cam_stop_flag:
            cam_stop_flag.set()
        mic_stream.stop(); mic_stream.close()
        net_t.join(timeout=2.0)
        if cam_proc:
            cam_proc.join(timeout=2.0)
            if cam_proc.is_alive():
                cam_proc.terminate()
        if cam_bridge_t:
            cam_bridge_t.join(timeout=2.0)

    app.aboutToQuit.connect(_shutdown)

    try:
        if hasattr(pg, 'exec'):
            pg.exec()
        else:
            app.exec_()
    except KeyboardInterrupt:
        pass
    finally:
        print('\n[RUN] Stopping...')
        _shutdown()


if __name__ == '__main__':
    main()