"""
experiment_collector.py
────────────────────────────────────────────────────────────────────────────
WristPad data-collection app — two windows, one process.

  INSTRUCTOR window: everything from realtime_multimodal_viz.py (live
  waveforms/spectrograms/IMU/camera, touch ON/OFF indicator, threshold
  tuning) PLUS recording controls: a REC toggle (spacebar or button —
  press once to start, press again to stop; no more holding it down),
  material presets that set the touch-detection band-pass to a
  known-good range for that surface, and a label field for the current
  stimulus/dataset label.

  EXPERIMENTER window: a second, plain window meant for a second monitor
  facing the person writing — shows the current stimulus in large text,
  a "Writing start! / Writing end — please wait" banner driven directly
  by the instructor's REC state, and a free-text instruction line the
  instructor can push over.

Data pipeline (unchanged from unified_collector.py): watch TCP, camera/
MediaPipe in a separate process, surface-mic touch detection (band-pass →
median filter → attack/release envelope → dB-above-adaptive-floor →
threshold+hysteresis → minimum-duration debounce, all in the audio
thread), RTBGN/RTEND two-point watch-clock resync, rolling IMU/watch-audio
buffers for post-hoc-accurate trial cropping.

Trial boundaries — unchanged two-tier design:
  REC toggle = coarse gate (was: holding spacebar). Within it, one trial
  is saved per REC session, trimmed to [first audio touch-on, last audio
  touch-off] observed while REC was on — not the raw REC start/stop times,
  which still carry human reaction delay. Every individual touch-on/off
  transition is logged to events.csv regardless (rec_start/rec_end and
  audio_touch_on/audio_touch_off, all with precise, debounce-backdated
  timestamps) — nothing about the raw detections is discarded, only the
  *grouping* into one saved trial per REC session is coarser than the raw
  event log. If the detector never fires during a whole REC session, that
  session is saved as one legacy-style trial spanning REC start/stop
  instead, so a detector problem never means silently losing a take.

Requires: pyqtgraph, PyQt5 (or PySide2/PyQt6/PySide6), pynput, sounddevice,
numpy, scipy, opencv-python, fingertip_imu_multi.py on PYTHONPATH.

Usage:
    python experiment_collector.py
    python experiment_collector.py --mic-device 1 --mic-channel 1
    python experiment_collector.py --dataset-root dataset/ --list-devices
"""

import argparse
import csv
import json
import multiprocessing as mp
import queue
import signal
import socket
import sys
import threading
import time
import wave
from collections import deque
from datetime import datetime
from math import gcd
from pathlib import Path

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from pynput import keyboard
from scipy.signal import butter, sosfilt, sosfilt_zi, resample_poly

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
# NOTE: cv2 is imported lazily inside _camera_process_fn (the child process).

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=False)
pg.setConfigOptions(imageAxisOrder='row-major')

_AXIS_COLORS = {'x': '#d62728', 'y': '#2ca02c', 'z': '#1f77b4'}

# ─── Config ───────────────────────────────────────────────────────────────────
WATCH_HOST       = '0.0.0.0'
WATCH_PORT       = 50005
WATCH_AUDIO_SR   = 48000
WATCH_FRAME_SIZE = WATCH_AUDIO_SR // 25
WATCH_BUF_SIZE   = WATCH_FRAME_SIZE * 2

# See unified_collector.py's original note: the watch timestamps an audio
# frame only once it's fully buffered, so watch_ts_ms is systematically late
# relative to true capture time. Corrected once, at trial-crop time, after
# the RTBGN-based watch-clock -> PC-time mapping is known.
WATCH_AUDIO_LATENCY_SEC = 0.045

MIC_SR         = 192000
MIC_CHANNELS   = 4
MIC_TARGET_CH  = 1
MIC_BLOCK_SIZE = 512
MIC_GAIN       = 1.0

CAM_SMOOTHING_WINDOW = 3
CAM_EMA_ALPHA        = 0.2
CAM_FLUSH_EVERY_N    = 10

DATA_ROOT      = Path('data')
SESSION_PREFIX = 'session'

FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']

# Material -> (band_low_hz, band_high_hz) for the touch-detection band-pass.
MATERIAL_PRESETS = {
    'wood':    (3000.0, 6000.0),
    'paper':   (3000.0, 6000.0),
    'fabric':  (3000.0, 6000.0),
    'acrylic': (3000.0, 6000.0),
}

stop_event = threading.Event()


# ─── Global session/file state (unchanged from unified_collector.py) ────────
_lock = threading.Lock()

_session_dir:   Path | None = None
_session_start: float | None = None

_watch_wf:   wave.Wave_write | None = None
_mic_wf:     wave.Wave_write | None = None
_imu_fp      = None
_imu_writer  = None
_imu_flush_n = 0

_cam_fp      = None
_cam_writer  = None
_cam_flush_n = 0

_events_fp     = None
_events_writer = None
_space_down    = False

_watch_audio_frames_fp:     None = None
_watch_audio_frames_writer: None = None
_watch_audio_session_samples: int = 0

_trial_lock       = threading.Lock()
_trial_active     = False
_trial_start_offset: float | None = None
_trial_buffers = {
    'fingertip': [],
    'mic':       [],
}
_trial_queue: "queue.Queue" = queue.Queue()
_mic_sr_runtime: int = 192000

_rec_active: bool = False
_audio_touch_start: float | None = None
_audio_first_on_offset: float | None = None
_audio_last_off_offset: float | None = None
_audio_trial_margin: float = 0.0
_current_label: str = ''

_pending_lock:   threading.Lock = threading.Lock()
_pending_starts: list = []

ROLLING_RETENTION_SEC = 30.0
IMU_GRACE_SEC         = 0.5
WATCH_AUDIO_GRACE_SEC = 0.5

_rolling_lock: threading.Lock = threading.Lock()
_imu_rolling:         "deque" = deque()
_watch_audio_rolling: "deque" = deque()

_trial_dataset_root: Path = Path('dataset')
_trial_margin:       float = 0.1

_watch_audio_offset: float | None = None
_mic_offset:         float | None = None
_imu_offset:         float | None = None
_cam_offset:         float | None = None
_sync: dict = {}

_watch_rms: float = 0.0
_mic_rms:   float = 0.0

_mic_sos = butter(2, 10.0 / (MIC_SR / 2), btype='high', output='sos')
_mic_zi  = sosfilt_zi(_mic_sos) * 0
_mic_target_ch = 0


# ─── Utilities ───────────────────────────────────────────────────────────────
def _offset() -> float:
    if _session_start is None:
        return 0.0
    return time.perf_counter() - _session_start


_verbose = False


def _log(tag: str, msg: str):
    print(f'[{_offset():8.3f}s][{tag}] {msg}')


def _rolling_cutoff(now_ts: float) -> float:
    age_based_cutoff = now_ts - ROLLING_RETENTION_SEC
    with _pending_lock:
        if _pending_starts:
            return min(age_based_cutoff, min(_pending_starts))
    return age_based_cutoff


# ─── Session management ───────────────────────────────────────────────────────
def start_session(label: str = '') -> Path:
    global _session_dir, _session_start
    global _watch_wf, _mic_wf, _imu_fp, _imu_writer
    global _cam_fp, _cam_writer
    global _events_fp, _events_writer
    global _watch_audio_frames_fp, _watch_audio_frames_writer, _watch_audio_session_samples
    global _watch_audio_offset, _mic_offset, _imu_offset, _cam_offset, _sync

    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    name   = f'{SESSION_PREFIX}_{ts_str}' + (f'_{label}' if label else '')
    _session_dir   = DATA_ROOT / name
    _session_dir.mkdir(parents=True, exist_ok=True)
    _session_start = time.perf_counter()

    _watch_audio_offset = None
    _mic_offset         = None
    _imu_offset         = None
    _cam_offset         = None

    _watch_wf = wave.open(str(_session_dir / 'watch_audio.wav'), 'wb')
    _watch_wf.setnchannels(1); _watch_wf.setsampwidth(2); _watch_wf.setframerate(WATCH_AUDIO_SR)

    _mic_wf = wave.open(str(_session_dir / 'surface_mic.wav'), 'wb')
    _mic_wf.setnchannels(1); _mic_wf.setsampwidth(2); _mic_wf.setframerate(MIC_SR)

    _imu_fp     = open(_session_dir / 'imu.csv', 'w', newline='')
    _imu_writer = csv.writer(_imu_fp)
    _imu_writer.writerow(['timestamp_sec', 'sensor', 'v1', 'v2', 'v3', 'watch_ts_ms'])

    _cam_fp     = open(_session_dir / 'fingertip_imu.csv', 'w', newline='')
    _cam_writer = csv.writer(_cam_fp)
    _cam_writer.writerow([
        'timestamp_sec', 'finger', 'hand_label', 'detected',
        'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'pos_x', 'pos_y', 'pos_z',
    ])

    _events_fp     = open(_session_dir / 'events.csv', 'w', newline='')
    _events_writer = csv.writer(_events_fp)
    _events_writer.writerow(['timestamp_sec', 'event'])

    _watch_audio_frames_fp     = open(_session_dir / 'watch_audio_frames.csv', 'w', newline='')
    _watch_audio_frames_writer = csv.writer(_watch_audio_frames_fp)
    _watch_audio_frames_writer.writerow(['sample_offset', 'num_samples', 'watch_ts_ms'])
    _watch_audio_session_samples = 0

    _sync = {
        'session_start_epoch':      time.time(),
        'label':                    label,
        'watch_audio_sr':           WATCH_AUDIO_SR,
        'surface_mic_sr':           MIC_SR,
        'watch_audio_offset_sec':   None,
        'surface_mic_offset_sec':   None,
        'imu_offset_sec':           None,
        'fingertip_imu_offset_sec': None,
        'rtbgn_watch_ms': None, 'rtbgn_pc_sec': None,
        'rtend_watch_ms': None, 'rtend_pc_sec': None,
    }
    print(f'[SESSION] Started -> {_session_dir}')
    return _session_dir


def close_session():
    global _watch_wf, _mic_wf, _imu_fp, _cam_fp, _events_fp, _watch_audio_frames_fp

    for wf in [_watch_wf, _mic_wf]:
        if wf:
            wf.close()
    _watch_wf = _mic_wf = None
    if _imu_fp: _imu_fp.close(); _imu_fp = None
    if _cam_fp: _cam_fp.close(); _cam_fp = None
    if _events_fp: _events_fp.close(); _events_fp = None
    if _watch_audio_frames_fp: _watch_audio_frames_fp.close(); _watch_audio_frames_fp = None

    if _session_dir:
        with open(_session_dir / 'sync.json', 'w') as f:
            json.dump(_sync, f, indent=2)
        print(f'\n[SESSION] Saved -> {_session_dir}')
        _check_watch_connection_quality()
        _recalibrate_session_trials(_session_dir, _trial_dataset_root)


def _check_watch_connection_quality():
    rtbgn_watch_ms = _sync.get('rtbgn_watch_ms'); rtbgn_pc_sec = _sync.get('rtbgn_pc_sec')
    rtend_watch_ms = _sync.get('rtend_watch_ms'); rtend_pc_sec = _sync.get('rtend_pc_sec')
    if not (rtbgn_watch_ms and rtbgn_pc_sec and rtend_watch_ms and rtend_pc_sec):
        return
    watch_elapsed = (rtend_watch_ms - rtbgn_watch_ms) / 1000.0
    pc_elapsed    = rtend_pc_sec - rtbgn_pc_sec
    if pc_elapsed <= 0:
        return
    ratio = watch_elapsed / pc_elapsed
    print(f'[QUALITY] watch-clock elapsed={watch_elapsed:.2f}s  PC-clock elapsed={pc_elapsed:.2f}s  ratio={ratio:.2%}')


def _read_wav_samples(path: Path):
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate(); n = wf.getnframes()
        data = np.frombuffer(wf.readframes(n), dtype='<i2')
    return sr, data


def _recalibrate_session_trials(session_dir: Path, dataset_root: Path):
    sync_path = session_dir / 'sync.json'
    if not sync_path.exists():
        return
    with open(sync_path) as f:
        sync = json.load(f)

    rtbgn_watch_ms = sync.get('rtbgn_watch_ms'); rtbgn_pc_sec = sync.get('rtbgn_pc_sec')
    rtend_watch_ms = sync.get('rtend_watch_ms'); rtend_pc_sec = sync.get('rtend_pc_sec')
    if not (rtbgn_watch_ms and rtbgn_pc_sec and rtend_watch_ms and rtend_pc_sec):
        return

    watch_span_sec = (rtend_watch_ms - rtbgn_watch_ms) / 1000.0
    pc_span_sec    = rtend_pc_sec - rtbgn_pc_sec
    if watch_span_sec <= 0 or pc_span_sec <= 0:
        return
    rate = pc_span_sec / watch_span_sec

    def aligned_pc(watch_ts_ms: float) -> float:
        return rtbgn_pc_sec + (watch_ts_ms - rtbgn_watch_ms) / 1000.0 * rate

    metadata_path = dataset_root / 'metadata.csv'
    if not metadata_path.exists():
        return
    session_name = session_dir.name
    with open(metadata_path) as f:
        trials = [row for row in csv.DictReader(f) if row['session'] == session_name]
    if not trials:
        return

    frames_path = session_dir / 'watch_audio_frames.csv'
    wav_path    = session_dir / 'watch_audio.wav'
    imu_path    = session_dir / 'imu.csv'

    frame_rows = []; wav_samples = None
    if frames_path.exists() and wav_path.exists():
        with open(frames_path) as f:
            frame_rows = list(csv.DictReader(f))
        _, wav_samples = _read_wav_samples(wav_path)

    imu_rows = []
    if imu_path.exists():
        with open(imu_path) as f:
            imu_rows = list(csv.DictReader(f))

    for trial in trials:
        label = trial['label']; trial_idx = int(trial['trial_idx'])
        trial_start = float(trial['start_sec']); trial_end = float(trial['end_sec'])
        trial_dir = dataset_root / label / f'trial_{trial_idx:03d}'
        if not trial_dir.exists():
            continue

        if frame_rows and wav_samples is not None:
            selected = []
            for fr in frame_rows:
                wts_str = fr.get('watch_ts_ms', '')
                if not wts_str:
                    continue
                apc = aligned_pc(float(wts_str)) - WATCH_AUDIO_LATENCY_SEC
                if trial_start <= apc <= trial_end:
                    start_i = int(fr['sample_offset']); n = int(fr['num_samples'])
                    selected.append((apc, wav_samples[start_i:start_i + n]))
            if selected:
                selected.sort(key=lambda item: item[0])
                corrected = np.concatenate([chunk for _, chunk in selected])
                wavfile.write(trial_dir / 'watch_audio.wav', WATCH_AUDIO_SR, corrected)


# ─── Display ring buffers ─────────────────────────────────────────────────────
class ScrollingWaveform:
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
        n = len(self.buf)
        if n == 0:
            return np.array([0.0]), np.array([0.0])
        y = np.asarray(self.buf, dtype=np.float32)
        x = (np.arange(n) - (n - 1)) * self.dt
        return x, y


class ScrollingSpectrogram:
    def __init__(self, sr: float, n_fft: int = 1024, hop_sec: float = 0.02,
                 max_cols: int = 300, freq_max: float | None = None, db_floor: float = -100.0):
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
    def __init__(self, window_sec: float, expected_hz: float = 100.0):
        maxlen = max(4, int(window_sec * expected_hz))
        self.t = deque(maxlen=maxlen); self.x = deque(maxlen=maxlen)
        self.y = deque(maxlen=maxlen); self.z = deque(maxlen=maxlen)
        self._t0 = None

    def push(self, v1, v2, v3, ts: float | None = None):
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
        if len(self.t) < 2:
            return 0.0
        span = self.t[-1] - self.t[0]
        return (len(self.t) - 1) / span if span > 0 else 0.0


disp_surface_wave: ScrollingWaveform | None = None
disp_surface_spec: ScrollingSpectrogram | None = None
disp_watch_wave:   ScrollingWaveform | None = None
disp_watch_spec:   ScrollingSpectrogram | None = None
disp_watch_acc  = ScrollingIMU(window_sec=5.0, expected_hz=100.0)
disp_watch_gyro = ScrollingIMU(window_sec=5.0, expected_hz=100.0)
disp_finger_acc  = ScrollingIMU(window_sec=5.0, expected_hz=30.0)
disp_finger_gyro = ScrollingIMU(window_sec=5.0, expected_hz=30.0)
_display_finger = 'index'


# ─── Stream writers ───────────────────────────────────────────────────────────
def _write_watch_audio(raw_bytes: bytes, watch_ts_ms: float | None = None, arrival_offset: float | None = None):
    global _watch_audio_offset, _watch_rms, _watch_audio_session_samples
    ts = arrival_offset if arrival_offset is not None else _offset()
    samples = np.frombuffer(raw_bytes, dtype='<i2').astype(np.float32)
    _watch_rms = float(np.sqrt(np.mean(samples ** 2)))
    with _lock:
        if _watch_audio_offset is None:
            _watch_audio_offset = ts
            _sync['watch_audio_offset_sec'] = _watch_audio_offset
    if _watch_wf:
        _watch_wf.writeframes(raw_bytes)

    n_samples = len(raw_bytes) // 2
    if _watch_audio_frames_writer:
        _watch_audio_frames_writer.writerow([
            _watch_audio_session_samples, n_samples,
            f'{watch_ts_ms:.0f}' if watch_ts_ms else '',
        ])
    _watch_audio_session_samples += n_samples

    with _rolling_lock:
        _watch_audio_rolling.append((ts, raw_bytes, watch_ts_ms))
        cutoff = _rolling_cutoff(ts)
        while _watch_audio_rolling and _watch_audio_rolling[0][0] < cutoff:
            _watch_audio_rolling.popleft()

    if disp_watch_wave is not None:
        norm = samples / 32768.0
        disp_watch_wave.push(norm)
        disp_watch_spec.push(norm)


def _write_imu(sensor: str, v1: float, v2: float, v3: float, watch_ts_ms: float = 0.0,
               display_ts: float | None = None, arrival_offset: float | None = None):
    global _imu_offset, _imu_flush_n
    ts = arrival_offset if arrival_offset is not None else _offset()
    with _lock:
        if _imu_offset is None:
            _imu_offset = ts
            _sync['imu_offset_sec'] = _imu_offset
        if _imu_writer:
            _imu_writer.writerow([f'{ts:.6f}', sensor, f'{v1:.6f}', f'{v2:.6f}', f'{v3:.6f}', f'{watch_ts_ms:.0f}'])
            _imu_flush_n += 1
            if _imu_flush_n % 40 == 0 and _imu_fp:
                _imu_fp.flush()

    with _rolling_lock:
        _imu_rolling.append((ts, sensor, v1, v2, v3, watch_ts_ms))
        cutoff = _rolling_cutoff(ts)
        while _imu_rolling and _imu_rolling[0][0] < cutoff:
            _imu_rolling.popleft()

    target = disp_watch_acc if sensor == 'acc' else disp_watch_gyro
    target.push(v1, v2, v3, ts=display_ts)


def _write_fingertip_imu(records: list):
    global _cam_offset, _cam_flush_n
    with _lock:
        if _cam_offset is None:
            _cam_offset = _offset()
            _sync['fingertip_imu_offset_sec'] = _cam_offset
        if _cam_writer:
            for r in records:
                _cam_writer.writerow([
                    f'{r.timestamp:.6f}', r.finger, r.hand_label, int(r.detected),
                    f'{r.accel_x:.4f}', f'{r.accel_y:.4f}', f'{r.accel_z:.4f}',
                    f'{r.gyro_x:.6f}', f'{r.gyro_y:.6f}', f'{r.gyro_z:.6f}',
                    f'{r.pos_x:.3f}', f'{r.pos_y:.3f}', f'{r.pos_z:.3f}',
                ])
            _cam_flush_n += 1
            if _cam_flush_n % CAM_FLUSH_EVERY_N == 0 and _cam_fp:
                _cam_fp.flush()

    with _trial_lock:
        if _trial_active:
            _trial_buffers['fingertip'].extend(records)

    for r in records:
        if r.finger != _display_finger or not r.detected:
            continue
        acc_vals = (r.accel_x, r.accel_y, r.accel_z)
        gyro_vals = (r.gyro_x, r.gyro_y, r.gyro_z)
        if all(np.isfinite(v) for v in acc_vals):
            disp_finger_acc.push(*acc_vals)
        if all(np.isfinite(v) for v in gyro_vals):
            disp_finger_gyro.push(*gyro_vals)


# ─── Touch detection state with Calibration Phase ─────────────────────────────
ENV_ATTACK_TAU_SEC  = 0.005
ENV_RELEASE_TAU_SEC = 0.08

_touch_band_low_hz  = MATERIAL_PRESETS['wood'][0]
_touch_band_high_hz = MATERIAL_PRESETS['wood'][1]
_current_material   = 'wood'

_mic_band_sos = None
_mic_band_zi  = None
_envelope = 0.0
_noise_floor = None
_noise_floor_db_abs = -100.0
_touch_metric_db = -60.0

# 캘리브레이션 관련 상태 변수
_is_calibrating = True
_calibration_duration_sec = 1.5   # 시작 시 조용히 대기할 시간 (초)
_calibration_start_time = None
_calibration_samples = []

_touch_median_buf: "deque | None" = None
_touch_on_threshold_db  = 8.0
_touch_off_threshold_db = 5.0
_touch_min_on_sec  = 0.03
_touch_min_off_sec = 0.1
_touch_on_state = False
_touch_candidate_on_time  = 0.0
_touch_candidate_off_time = 0.0

_material_change_queue: "queue.Queue" = queue.Queue(maxsize=1)


def _rebuild_touch_band_filter(mic_sr: int, band_low: float, band_high: float):
    global _mic_band_sos, _mic_band_zi, _touch_band_low_hz, _touch_band_high_hz
    global _envelope, _noise_floor, _noise_floor_db_abs, _touch_metric_db, _touch_on_state
    global _touch_candidate_on_time, _touch_candidate_off_time
    global _is_calibrating, _calibration_start_time, _calibration_samples

    nyquist = mic_sr / 2.0
    band_high = min(band_high, nyquist - 100.0)
    if band_high <= band_low:
        _mic_band_sos = None
        print(f'[TOUCH] mic-sr={mic_sr} too low for band — touch detection disabled.')
        return
    
    _mic_band_sos = butter(4, [band_low / nyquist, band_high / nyquist], btype='band', output='sos')
    
    # 올바른 SOS 구조체 모양(4, 2)에 맞추어 zi 초기화
    _mic_band_zi = sosfilt_zi(_mic_band_sos) * 0  
    
    _touch_band_low_hz, _touch_band_high_hz = band_low, band_high
    _envelope = 0.0
    _noise_floor = None
    _noise_floor_db_abs = -100.0
    _touch_metric_db = -60.0
    _touch_on_state = False
    _touch_candidate_on_time = 0.0
    _touch_candidate_off_time = 0.0
    
    _is_calibrating = True
    _calibration_start_time = time.perf_counter()
    _calibration_samples = []

    if _touch_median_buf is not None:
        _touch_median_buf.clear()
    print(f'[TOUCH] band-pass set to {band_low:.0f}-{band_high:.0f}Hz — Calibrating floor for {_calibration_duration_sec}s...')


def set_material(name: str):
    if name not in MATERIAL_PRESETS:
        return
    band_low, band_high = MATERIAL_PRESETS[name]
    try:
        _material_change_queue.get_nowait()
    except queue.Empty:
        pass
    _material_change_queue.put_nowait((name, band_low, band_high))


def _sync_touch_thresholds(on_db: float, hyst_db: float):
    global _touch_on_threshold_db, _touch_off_threshold_db
    off_db = on_db - hyst_db
    _touch_on_threshold_db = on_db
    _touch_off_threshold_db = off_db


def _write_event(event: str):
    ts = _offset()
    with _lock:
        if _events_writer:
            _events_writer.writerow([f'{ts:.6f}', event])
            _events_fp.flush()
    _log('EVENT', f'{event}  offset={ts:.4f}s')


def _write_event_at(event: str, ts: float):
    with _lock:
        if _events_writer:
            _events_writer.writerow([f'{ts:.6f}', event])
            _events_fp.flush()
    _log('EVENT', f'{event}  offset={ts:.4f}s (backdated)')


def toggle_recording():
    global _trial_active, _trial_start_offset, _rec_active, _current_label, _current_stimulus
    global _audio_touch_start, _audio_first_on_offset, _audio_last_off_offset
    global _touch_candidate_on_time, _touch_candidate_off_time

    if not _rec_active:
        _current_label = _label_getter() if _label_getter else ''
        _current_stimulus = _current_label
        _write_event('rec_start')
        with _trial_lock:
            for k in _trial_buffers:
                _trial_buffers[k].clear()
            _trial_start_offset = _offset()
            _trial_active = True
        _rec_active = True
        _audio_first_on_offset = None
        _audio_last_off_offset = None
        _touch_candidate_on_time = 0.0
        _touch_candidate_off_time = 0.0
        if _touch_on_state:
            _audio_touch_start = _offset()
            _audio_first_on_offset = _audio_touch_start
            _write_event_at('audio_touch_on', _audio_touch_start)
    else:
        release_offset = _offset()
        if _audio_touch_start is not None:
            _write_event_at('audio_touch_off', release_offset)
            _audio_last_off_offset = release_offset
            _audio_touch_start = None
        _rec_active = False

        with _trial_lock:
            _trial_active = False
            rec_start = _trial_start_offset
            snapshot = {k: list(v) for k, v in _trial_buffers.items()}
        _write_event('rec_end')

        label = _current_label
        if _audio_first_on_offset is not None and _audio_last_off_offset is not None \
                and _audio_last_off_offset > _audio_first_on_offset:
            with _pending_lock:
                _pending_starts.append(_audio_first_on_offset)
            _trial_queue.put((_audio_first_on_offset, _audio_last_off_offset, snapshot, 'audio', label))
        elif rec_start is not None and release_offset > rec_start:
            with _pending_lock:
                _pending_starts.append(rec_start)
            _trial_queue.put((rec_start, release_offset, snapshot, 'spacebar_fallback', label))

        _audio_first_on_offset = None
        _audio_last_off_offset = None


_label_getter = None


def _on_key_press(key):
    global _space_down
    if key == keyboard.Key.space and not _space_down:
        _space_down = True
        toggle_recording()
    elif key == keyboard.Key.esc:
        stop_event.set()
        return False


def _on_key_release(key):
    global _space_down
    if key == keyboard.Key.space:
        _space_down = False


def _trial_worker_fn():
    while not stop_event.is_set() or not _trial_queue.empty():
        try:
            start, end, snapshot, trigger, label = _trial_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            time.sleep(max(IMU_GRACE_SEC, WATCH_AUDIO_GRACE_SEC))

            with _rolling_lock:
                snapshot['imu']         = list(_imu_rolling)
                snapshot['watch_audio'] = list(_watch_audio_rolling)

            with _pending_lock:
                if start in _pending_starts:
                    _pending_starts.remove(start)

            process_trial(start, end, snapshot, trigger, label)
        except Exception as e:
            _log('TRIAL', f'error while processing: {e}')


def process_trial(start: float, end: float, snapshot: dict, trigger: str, label: str):
    margin = _audio_trial_margin if trigger == 'audio' else _trial_margin
    trial_start = start + margin
    trial_end   = end - margin
    if trial_end <= trial_start:
        return

    rtbgn_watch_ms = _sync.get('rtbgn_watch_ms')
    rtbgn_pc_sec   = _sync.get('rtbgn_pc_sec')
    use_rtbgn = bool(rtbgn_watch_ms and rtbgn_pc_sec)

    imu_rel = []
    for (ts, sensor, v1, v2, v3, watch_ts_ms) in snapshot['imu']:
        if use_rtbgn and watch_ts_ms:
            aligned_pc = (watch_ts_ms - rtbgn_watch_ms) / 1000.0 + rtbgn_pc_sec
        else:
            aligned_pc = ts
        if trial_start <= aligned_pc <= trial_end:
            imu_rel.append((aligned_pc - trial_start, sensor, v1, v2, v3))

    ft_rel = [
        (r.timestamp - trial_start, r.finger, r.hand_label, int(r.detected),
         r.accel_x, r.accel_y, r.accel_z, r.gyro_x, r.gyro_y, r.gyro_z,
         r.pos_x, r.pos_y, r.pos_z)
        for r in snapshot['fingertip']
        if trial_start <= r.timestamp <= trial_end
    ]

    if not imu_rel and not ft_rel:
        return

    wa_pieces = []
    for (ts, chunk, watch_ts_ms) in snapshot['watch_audio']:
        if use_rtbgn and watch_ts_ms:
            frame_start_pc = (watch_ts_ms - rtbgn_watch_ms) / 1000.0 + rtbgn_pc_sec - WATCH_AUDIO_LATENCY_SEC
        else:
            frame_start_pc = ts - WATCH_AUDIO_LATENCY_SEC
        frame_samples = np.frombuffer(chunk, dtype='<i2')
        n = len(frame_samples)
        frame_end_pc = frame_start_pc + n / WATCH_AUDIO_SR
        if frame_end_pc < trial_start or frame_start_pc > trial_end:
            continue
        sample_times = frame_start_pc + np.arange(n) / WATCH_AUDIO_SR
        mask = (sample_times >= trial_start) & (sample_times <= trial_end)
        if np.any(mask):
            wa_pieces.append((frame_start_pc, frame_samples[mask]))
    wa_pieces.sort(key=lambda item: item[0])
    wa_samples = np.concatenate([s for _, s in wa_pieces]) if wa_pieces else np.array([], dtype=np.int16)

    mic_blocks = [blk for (ts, blk) in snapshot['mic'] if trial_start <= ts <= trial_end]
    if mic_blocks:
        mic_concat = np.concatenate(mic_blocks).astype(np.float64)
        if _mic_sr_runtime != WATCH_AUDIO_SR:
            g = gcd(WATCH_AUDIO_SR, _mic_sr_runtime)
            mic_rs = resample_poly(mic_concat, WATCH_AUDIO_SR // g, _mic_sr_runtime // g)
        else:
            mic_rs = mic_concat
    else:
        mic_rs = np.array([], dtype=np.float64)
    mic_int16 = np.clip(mic_rs * 32767, -32768, 32767).astype(np.int16)

    label_use = label if label else 'unlabeled'
    label_dir = _trial_dataset_root / label_use
    label_dir.mkdir(parents=True, exist_ok=True)
    trial_idx = len(sorted(label_dir.glob('trial_*'))) + 1
    trial_dir = label_dir / f'trial_{trial_idx:03d}'
    trial_dir.mkdir(parents=True, exist_ok=True)

    wavfile.write(trial_dir / 'watch_audio.wav', WATCH_AUDIO_SR, wa_samples)
    wavfile.write(trial_dir / 'surface_mic.wav', WATCH_AUDIO_SR, mic_int16)

    with open(trial_dir / 'imu.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_aligned', 'sensor', 'v1', 'v2', 'v3'])
        for ta, sensor, v1, v2, v3 in sorted(imu_rel, key=lambda row: row[0]):
            w.writerow([f'{ta:.6f}', sensor, f'{v1:.6f}', f'{v2:.6f}', f'{v3:.6f}'])

    with open(trial_dir / 'fingertip_imu.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_aligned', 'finger', 'hand_label', 'detected',
                     'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z',
                     'pos_x', 'pos_y', 'pos_z'])
        for row in ft_rel:
            ta, finger, hand_label, detected, ax, ay, az, gx, gy, gz, px, py, pz = row
            w.writerow([f'{ta:.6f}', finger, hand_label, detected,
                        f'{ax:.4f}', f'{ay:.4f}', f'{az:.4f}',
                        f'{gx:.6f}', f'{gy:.6f}', f'{gz:.6f}',
                        f'{px:.3f}', f'{py:.3f}', f'{pz:.3f}'])

    metadata_path = _trial_dataset_root / 'metadata.csv'
    write_header = not metadata_path.exists()
    with open(metadata_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['session', 'label', 'trial_idx', 'start_sec', 'end_sec',
                        'duration_sec', 'margin_sec', 'trigger', 'material'])
        w.writerow([
            _session_dir.name if _session_dir else '',
            label_use, trial_idx,
            f'{trial_start:.6f}', f'{trial_end:.6f}',
            f'{trial_end - trial_start:.6f}', margin, trigger, _current_material,
        ])


def _parse_imu_packet(pkt: bytes, sensor: str, arrival_pc: float, session_start: float) -> int:
    try:
        txt = pkt[5:].decode('utf-8', errors='ignore').strip()
    except Exception:
        return 0

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
        return 0

    arrival_offset = arrival_pc - session_start
    if len(samples) > 1 and all(s[3] is not None for s in samples):
        last_watch_sec = samples[-1][3] / 1000.0
        for v1, v2, v3, watch_ts_ms in samples:
            display_ts = arrival_pc - (last_watch_sec - watch_ts_ms / 1000.0)
            _write_imu(sensor, v1, v2, v3, watch_ts_ms or 0.0,
                       display_ts=display_ts, arrival_offset=arrival_offset)
    else:
        for v1, v2, v3, watch_ts_ms in samples:
            _write_imu(sensor, v1, v2, v3, watch_ts_ms or 0.0,
                       display_ts=arrival_pc, arrival_offset=arrival_offset)
    return len(samples)


# ─── Surface mic callback & Audio Worker ─────────────────────────────────────
_audio_process_queue: "queue.Queue" = queue.Queue()
_mic_wav_queue: "queue.Queue" = queue.Queue()


def _mic_callback(indata, frames, time_info, status):
    global _mic_offset, _mic_zi, _mic_rms
    if _session_dir is None:
        return
    ch  = MIC_TARGET_CH - 1
    raw = indata[:, ch].astype(np.float32)
    filtered, _mic_zi = sosfilt(_mic_sos, raw, zi=_mic_zi)
    amplified = np.clip(filtered * MIC_GAIN, -1.0, 1.0)
    _mic_rms = float(np.sqrt(np.mean(amplified ** 2)))

    with _lock:
        if _mic_offset is None:
            _mic_offset = _offset()
            _sync['surface_mic_offset_sec'] = _mic_offset

    try:
        _mic_wav_queue.put_nowait((amplified * 32767).astype(np.int16).tobytes())
    except Exception:
        pass

    with _trial_lock:
        if _trial_active:
            _trial_buffers['mic'].append((_offset(), amplified.copy()))

    try:
        _audio_process_queue.put_nowait((raw, amplified, frames, time.perf_counter()))
    except Exception:
        pass


def _mic_wav_writer_fn():
    while not stop_event.is_set():
        try:
            chunk = _mic_wav_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            if _mic_wf:
                _mic_wf.writeframes(chunk)
        except Exception:
            pass


def _audio_worker_fn():
    global _mic_band_zi, _envelope, _noise_floor, _noise_floor_db_abs, _touch_metric_db
    global _touch_on_state, _touch_candidate_on_time, _touch_candidate_off_time
    global _audio_touch_start, _audio_first_on_offset, _audio_last_off_offset
    global _current_material, _touch_on_threshold_db, _touch_off_threshold_db
    global _is_calibrating, _calibration_start_time, _calibration_samples

    while not stop_event.is_set():
        try:
            name, band_low, band_high = _material_change_queue.get_nowait()
            _rebuild_touch_band_filter(_mic_sr_runtime, band_low, band_high)
            _current_material = name
        except queue.Empty:
            pass

        try:
            raw, amplified, frames, arrival_pc = _audio_process_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            if disp_surface_wave is not None:
                disp_surface_wave.push(amplified)
                disp_surface_spec.push(amplified)

            if _mic_band_sos is None:
                continue
            band, _mic_band_zi = sosfilt(_mic_band_sos, raw, zi=_mic_band_zi)
            block_energy_raw = float(np.sqrt(np.mean(band ** 2)))
            block_dt = frames / _mic_sr_runtime

            # ── 캘리브레이션 단계 수행 ──
            if _is_calibrating:
                if _calibration_start_time is None:
                    _calibration_start_time = time.perf_counter()
                
                _calibration_samples.append(block_energy_raw)
                elapsed = time.perf_counter() - _calibration_start_time

                if elapsed >= _calibration_duration_sec:
                    arr = np.array(_calibration_samples, dtype=np.float32)
                    _noise_floor = float(np.percentile(arr, 10))
                    if _noise_floor < 1e-8:
                        _noise_floor = 1e-8
                    
                    _noise_floor_db_abs = 20.0 * np.log10(_noise_floor)
                    _is_calibrating = False
                    print(f'[TOUCH] Calibration complete! Fixed noise floor abs = {_noise_floor_db_abs:.1f} dB')
                continue

            block_energy = block_energy_raw
            if _touch_median_buf is not None:
                _touch_median_buf.append(block_energy_raw)
                block_energy = float(np.median(_touch_median_buf))

            coef_attack  = float(np.exp(-block_dt / ENV_ATTACK_TAU_SEC))
            coef_release = float(np.exp(-block_dt / ENV_RELEASE_TAU_SEC))
            
            if block_energy > _envelope:
                _envelope = coef_attack * _envelope + (1.0 - coef_attack) * block_energy
            else:
                _envelope = coef_release * _envelope + (1.0 - coef_release) * block_energy

            _touch_on_threshold_db = 8.0
            _touch_off_threshold_db = 5.0

            _touch_metric_db = 20.0 * np.log10((_envelope + 1e-8) / (_noise_floor + 1e-8))

            now_pc = arrival_pc
            if not _touch_on_state:
                if _touch_metric_db >= _touch_on_threshold_db:
                    _touch_candidate_on_time += block_dt
                    if _touch_candidate_on_time >= _touch_min_on_sec:
                        _touch_on_state = True
                        true_on_pc = now_pc - _touch_candidate_on_time
                        _touch_candidate_on_time = 0.0
                        if _rec_active and _session_start is not None:
                            _audio_touch_start = true_on_pc - _session_start
                            _write_event_at('audio_touch_on', _audio_touch_start)
                            if _audio_first_on_offset is None:
                                _audio_first_on_offset = _audio_touch_start
                else:
                    _touch_candidate_on_time = 0.0
            else:
                if _touch_metric_db < _touch_off_threshold_db:
                    _touch_candidate_off_time += block_dt
                    if _touch_candidate_off_time >= _touch_min_off_sec:
                        _touch_on_state = False
                        true_off_pc = now_pc - _touch_candidate_off_time
                        _touch_candidate_off_time = 0.0
                        if _rec_active and _audio_touch_start is not None and _session_start is not None:
                            end_offset = true_off_pc - _session_start
                            _write_event_at('audio_touch_off', end_offset)
                            _audio_last_off_offset = end_offset
                            _audio_touch_start = None
                else:
                    _touch_candidate_off_time = 0.0
        except Exception as e:
            _log('TOUCH', f'error processing audio block: {e}')


# ─── Watch TCP listener & Camera Process ──────────────────────────────────────
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


_watch_audio_queue: "queue.Queue" = queue.Queue()
_watch_imu_queue:   "queue.Queue" = queue.Queue()


def _dispatch_watch_packet(pkt: bytes, arrival_pc: float):
    total = len(pkt)
    if total == WATCH_BUF_SIZE + 8 or total == WATCH_BUF_SIZE:
        _watch_audio_queue.put((pkt, arrival_pc))
        return

    try:
        hdr = pkt[:5].decode('utf-8', errors='ignore')
    except Exception:
        return

    if hdr == 'SUBID':
        pass
    elif hdr == 'RTBGN':
        pc_sec = arrival_pc - _session_start
        if len(pkt) >= 13:
            watch_ms = int.from_bytes(pkt[5:13], byteorder='big', signed=False)
            with _lock:
                _sync['rtbgn_watch_ms'] = watch_ms
                _sync['rtbgn_pc_sec'] = pc_sec
    elif hdr == 'RTEND':
        pc_sec = arrival_pc - _session_start
        if len(pkt) >= 13:
            watch_ms = int.from_bytes(pkt[5:13], byteorder='big', signed=False)
            with _lock:
                _sync['rtend_watch_ms'] = watch_ms
                _sync['rtend_pc_sec'] = pc_sec
        _watch_audio_queue.put(('__RTEND__', arrival_pc))
    elif hdr == 'SOUND':
        raw = pkt[10:]
        buf = np.frombuffer(raw[:len(raw)//2*2], dtype='<i2')
        _watch_audio_queue.put((buf.tobytes(), arrival_pc))
    elif hdr == 'IMUAC':
        _watch_imu_queue.put((pkt, 'acc', arrival_pc))
    elif hdr == 'IMUGY':
        _watch_imu_queue.put((pkt, 'gyro', arrival_pc))
    elif total > 0 and total % 2 == 0:
        _watch_audio_queue.put((pkt, arrival_pc))


def _watch_audio_worker_fn():
    global _watch_wf
    while not stop_event.is_set():
        try:
            item, arrival_pc = _watch_audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            if item == '__RTEND__':
                if _watch_wf:
                    _watch_wf.close()
                    _watch_wf = None
                continue

            pkt = item
            arrival_offset = arrival_pc - _session_start
            total = len(pkt)
            if total == WATCH_BUF_SIZE + 8:
                watch_ts_ms = int.from_bytes(pkt[:8], byteorder='big', signed=False)
                _write_watch_audio(pkt[8:], watch_ts_ms, arrival_offset=arrival_offset)
            else:
                _write_watch_audio(pkt, arrival_offset=arrival_offset)
        except Exception:
            pass


def _watch_imu_worker_fn():
    while not stop_event.is_set():
        try:
            pkt, sensor, arrival_pc = _watch_imu_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            _parse_imu_packet(pkt, sensor, arrival_pc, _session_start)
        except Exception:
            pass


def _net_thread_fn(watch_port: int):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((WATCH_HOST, watch_port))
    srv.listen(16)
    srv.settimeout(1.0)

    while not stop_event.is_set():
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            continue
        except Exception:
            continue

        conn.settimeout(1.0)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            while not stop_event.is_set():
                header = _recv_exact(conn, 4)
                if header is None:
                    break
                msg_len = int.from_bytes(header, byteorder='big', signed=False)
                if msg_len <= 0 or msg_len > 10_000_000:
                    break
                payload = _recv_exact(conn, msg_len)
                if payload is None:
                    break
                arrival_pc = time.perf_counter()
                _dispatch_watch_packet(payload, arrival_pc)
        except Exception:
            pass
        finally:
            conn.close()
    srv.close()


def _camera_process_fn(camera_index: int, camera_pitch_deg, camera_roll_deg: float,
                        session_start: float,
                        record_queue: "mp.Queue", stop_flag: "mp.Event"):
    import cv2 as _cv2
    from fingertip_imu_multi import MultiFingertipIMUTracker, gravity_vector_from_camera_tilt

    gravity_mm_s2 = None
    if camera_pitch_deg is not None:
        gravity_mm_s2 = gravity_vector_from_camera_tilt(camera_pitch_deg, camera_roll_deg)

    tracker = MultiFingertipIMUTracker(
        max_num_hands=1, smoothing_window=CAM_SMOOTHING_WINDOW,
        gravity_mm_s2=gravity_mm_s2, ema_alpha=CAM_EMA_ALPHA,
    )
    cap = _cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        record_queue.put(None)
        return

    while not stop_flag.is_set():
        success, frame = cap.read()
        if not success:
            continue
        frame = _cv2.flip(frame, 1)
        ts = time.perf_counter() - session_start
        records = tracker.update(frame, timestamp=ts)
        try:
            record_queue.put_nowait(records)
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
            continue
        try:
            _write_fingertip_imu(records)
        except Exception:
            pass


_current_instruction: str = ''
_current_stimulus: str = ''


# ─── Instructor window ───────────────────────────────────────────────────────
class InstructorWindow(QtWidgets.QMainWindow):
    def __init__(self, window_sec: float, has_camera: bool, use_opengl: bool = False):
        super().__init__()
        self.window_sec = window_sec
        self.has_camera = has_camera
        self._metric_min = None
        self._metric_max = None
        self._last_rec_shown = None
        self.setWindowTitle('WristPad — Instructor')

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        rec_row = QtWidgets.QHBoxLayout()
        self.rec_btn = QtWidgets.QPushButton('● START RECORDING')
        self.rec_btn.setStyleSheet('font-size: 16px; font-weight: bold; padding: 8px; '
                                    'background-color: #d62728; color: white;')
        self.rec_btn.clicked.connect(self._on_rec_clicked)
        rec_row.addWidget(self.rec_btn)
        rec_row.addWidget(QtWidgets.QLabel('  (or press spacebar — press once to start, again to stop)'))
        rec_row.addStretch(1)
        self.status_label = QtWidgets.QLabel('')
        self.status_label.setStyleSheet('font-size: 12px; color: #333;')
        rec_row.addWidget(self.status_label)
        outer.addLayout(rec_row)

        meta_row = QtWidgets.QHBoxLayout()
        meta_row.addWidget(QtWidgets.QLabel('surface material:'))
        self.material_label = QtWidgets.QLabel(f'[{_current_material}] {_touch_band_low_hz:.0f}-{_touch_band_high_hz:.0f}Hz')
        self.material_label.setStyleSheet('font-weight: bold; color: #1f6feb; padding: 2px 6px; '
                                           'background-color: #eef4ff; border-radius: 3px;')
        for name in MATERIAL_PRESETS:
            btn = QtWidgets.QPushButton(name)
            btn.clicked.connect(lambda checked=False, n=name: self._on_material_clicked(n))
            meta_row.addWidget(btn)
        meta_row.addWidget(self.material_label)
        meta_row.addSpacing(20)
        meta_row.addWidget(QtWidgets.QLabel('label / stimulus:'))
        self.label_edit = QtWidgets.QLineEdit()
        self.label_edit.setPlaceholderText('e.g. A, 5, line_horizontal ...')
        self.label_edit.setMaximumWidth(200)
        meta_row.addWidget(self.label_edit)
        meta_row.addStretch(1)
        outer.addLayout(meta_row)

        thr_row = QtWidgets.QHBoxLayout()
        thr_row.addWidget(QtWidgets.QLabel('touch threshold (dB above noise floor):'))
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setDecimals(1); self.threshold_spin.setRange(-10.0, 60.0)
        self.threshold_spin.setEnabled(False)
        self.threshold_spin.setValue(_touch_on_threshold_db)
        thr_row.addWidget(self.threshold_spin)
        thr_row.addWidget(QtWidgets.QLabel('hysteresis (dB):'))
        self.hysteresis_spin = QtWidgets.QDoubleSpinBox()
        self.hysteresis_spin.setDecimals(1); self.hysteresis_spin.setRange(0.0, 30.0)
        self.hysteresis_spin.setEnabled(False)
        self.hysteresis_spin.setValue(_touch_on_threshold_db - _touch_off_threshold_db)
        thr_row.addWidget(self.hysteresis_spin)
        thr_row.addStretch(1)
        outer.addLayout(thr_row)

        grid = QtWidgets.QGridLayout()
        outer.addLayout(grid)

        self.pw_surface_wave = self._make_waveform_plot('Surface mic — waveform')
        self.pw_watch_wave   = self._make_waveform_plot('Watch mic — waveform')
        self.pw_surface_spec, self.img_surface_spec = self._make_spec_plot(
            'Surface mic — spectrogram', disp_surface_spec)
        self.pw_watch_spec, self.img_watch_spec = self._make_spec_plot(
            'Watch mic — spectrogram', disp_watch_spec)
        self.pw_wacc,  self.curves_wacc  = self._make_imu_plot('Watch IMU — acc')
        self.pw_wgyro, self.curves_wgyro = self._make_imu_plot('Watch IMU — gyro')
        self.pw_facc,  self.curves_facc  = self._make_imu_plot(f'Fingertip IMU ({_display_finger}) — acc')
        self.pw_fgyro, self.curves_fgyro = self._make_imu_plot(f'Fingertip IMU ({_display_finger}) — gyro')

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

        cam_status_text = ('camera: tracking active' if has_camera else '--no-camera specified')
        self.cam_status_label = QtWidgets.QLabel(cam_status_text)
        self.cam_status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.cam_status_label.setStyleSheet('background-color: #333; color: white; font-size: 12px; padding: 6px;')

        self.touch_label = QtWidgets.QLabel()
        self.touch_label.setAlignment(QtCore.Qt.AlignCenter)
        font = self.touch_label.font(); font.setPointSize(24); font.setBold(True)
        self.touch_label.setFont(font)
        self._set_touch_visual(False, -60.0)

        right_col = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.cam_status_label)
        right_layout.addWidget(self.touch_label, 1)

        minmax_row = QtWidgets.QHBoxLayout()
        self.minmax_label = QtWidgets.QLabel('since reset — min=–  max=–')
        self.minmax_label.setStyleSheet('font-size: 13px; font-weight: bold; color: #222; '
                                         'background-color: #eee; padding: 4px; border-radius: 3px;')
        self.minmax_label.setAlignment(QtCore.Qt.AlignCenter)
        self.reset_minmax_btn = QtWidgets.QPushButton('reset min/max')
        self.reset_minmax_btn.clicked.connect(self._reset_minmax)
        minmax_row.addWidget(self.minmax_label, 1)
        minmax_row.addWidget(self.reset_minmax_btn)
        right_layout.addLayout(minmax_row)

        grid.addWidget(right_col, 0, 2, 4, 1)
        grid.setColumnStretch(0, 1); grid.setColumnStretch(1, 1); grid.setColumnStretch(2, 1)
        self.resize(1650, 1050)

    def _make_waveform_plot(self, title: str) -> pg.PlotWidget:
        pw = pg.PlotWidget(title=title)
        pw.setLabel('bottom', 'time relative to now (s)'); pw.setLabel('left', 'amplitude')
        pw.setXRange(-self.window_sec, 0, padding=0); pw.setYRange(-1.05, 1.05, padding=0)
        pw.showGrid(x=True, y=True, alpha=0.25)
        curve = pw.plot(pen=pg.mkPen('#333333', width=1))
        curve.setDownsampling(auto=True, method='peak'); curve.setClipToView(True)
        pw._curve = curve
        return pw

    def _make_spec_plot(self, title: str, spec: "ScrollingSpectrogram"):
        pw = pg.PlotWidget(title=title)
        pw.setLabel('bottom', 'time relative to now (s)'); pw.setLabel('left', 'frequency (Hz)')
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
        pw.setXRange(-self.window_sec, 0, padding=0); pw.setYRange(0, freq_max, padding=0)
        return pw, img

    def _make_imu_plot(self, title: str, window_sec: float = 5.0):
        pw = pg.PlotWidget(title=title)
        pw.setLabel('bottom', 'time relative to now (s)'); pw.setLabel('left', 'value')
        pw.setXRange(-window_sec, 0, padding=0); pw.showGrid(x=True, y=True, alpha=0.25)
        pw.addLegend(offset=(5, 5))
        curves = {}
        for axis_name in ('x', 'y', 'z'):
            c = pw.plot(pen=pg.mkPen(_AXIS_COLORS[axis_name], width=1), name=axis_name)
            c.setDownsampling(auto=True, method='peak'); c.setClipToView(True)
            curves[axis_name] = c
        return pw, curves

    def _on_rec_clicked(self):
        toggle_recording()

    def _on_material_clicked(self, name: str):
        set_material(name)
        low, high = MATERIAL_PRESETS[name]
        self.material_label.setText(f'[{name}] {low:.0f}-{high:.0f}Hz')
        self._reset_minmax()

    def _reset_minmax(self):
        self._metric_min = None
        self._metric_max = None

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
        curves['x'].setData(t, x); curves['y'].setData(t, y); curves['z'].setData(t, z)
        if len(t) <= 1:
            return
        allv = np.concatenate([x, y, z])
        finite = allv[np.isfinite(allv)]
        if finite.size == 0:
            return
        lo, hi = float(np.min(finite)), float(np.max(finite))
        pad = max((hi - lo) * 0.15, 1e-3)
        pw.setYRange(lo - pad, hi + pad, padding=0)

    def _set_touch_visual(self, is_on: bool, metric_db: float):
        if _is_calibrating:
            self.touch_label.setStyleSheet('background-color: #f1c40f; color: black;')
            self.touch_label.setText('CALIBRATING...\nKeep surface quiet')
        elif is_on:
            self.touch_label.setStyleSheet('background-color: #2ca02c; color: white;')
            self.touch_label.setText(f'TOUCH ON\n({metric_db:.1f} dB above floor)')
        else:
            self.touch_label.setStyleSheet('background-color: #d62728; color: white;')
            self.touch_label.setText(f'TOUCH OFF\n({metric_db:.1f} dB above floor)')

    def update(self):
        self._update_waveform(self.pw_surface_wave, disp_surface_wave)
        self._update_waveform(self.pw_watch_wave, disp_watch_wave)
        self._update_spec(self.img_surface_spec, disp_surface_spec)
        self._update_spec(self.img_watch_spec, disp_watch_spec)
        self._update_imu(self.pw_wacc, self.curves_wacc, disp_watch_acc)
        self._update_imu(self.pw_wgyro, self.curves_wgyro, disp_watch_gyro)
        self._update_imu(self.pw_facc, self.curves_facc, disp_finger_acc)
        self._update_imu(self.pw_fgyro, self.curves_fgyro, disp_finger_gyro)
        self._set_touch_visual(_touch_on_state, _touch_metric_db)

        self.threshold_spin.setValue(_touch_on_threshold_db)
        self.hysteresis_spin.setValue(_touch_on_threshold_db - _touch_off_threshold_db)

        if np.isfinite(_touch_metric_db) and not _is_calibrating:
            self._metric_min = _touch_metric_db if self._metric_min is None else min(self._metric_min, _touch_metric_db)
            self._metric_max = _touch_metric_db if self._metric_max is None else max(self._metric_max, _touch_metric_db)
        if self._metric_min is not None:
            self.minmax_label.setText(f'since reset — min={self._metric_min:.1f}dB  max={self._metric_max:.1f}dB')
        else:
            self.minmax_label.setText('since reset — min=–  max=–')

        if _rec_active != self._last_rec_shown:
            self._last_rec_shown = _rec_active
            if _rec_active:
                self.rec_btn.setText('■ STOP RECORDING')
                self.rec_btn.setStyleSheet('font-size: 16px; font-weight: bold; padding: 8px; '
                                            'background-color: #2ca02c; color: white;')
            else:
                self.rec_btn.setText('● START RECORDING')
                self.rec_btn.setStyleSheet('font-size: 16px; font-weight: bold; padding: 8px; '
                                            'background-color: #d62728; color: white;')

        if _is_calibrating:
            self.status_label.setText(f'STATUS: Calibrating noise floor... keep surface quiet.')
        else:
            self.status_label.setText(
                f'surface mic RMS={_mic_rms:.4f}    floor abs={_noise_floor_db_abs:.1f}dB    '
                f'touch metric={_touch_metric_db:.1f}dB    material={_current_material}  '
                f'[{_touch_band_low_hz:.0f}-{_touch_band_high_hz:.0f}Hz]')

    def closeEvent(self, event):
        stop_event.set()
        super().closeEvent(event)


# ─── Experimenter window ─────────────────────────────────────────────────────
class ExperimenterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('WristPad — Experimenter')
        central = QtWidgets.QWidget()
        central.setStyleSheet('background-color: white;')
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.stimulus_label = QtWidgets.QLabel('')
        self.stimulus_label.setAlignment(QtCore.Qt.AlignCenter)
        f = self.stimulus_label.font(); f.setPointSize(140); f.setBold(True)
        self.stimulus_label.setFont(f)
        layout.addWidget(self.stimulus_label, 3)

        self.banner_label = QtWidgets.QLabel('')
        self.banner_label.setAlignment(QtCore.Qt.AlignCenter)
        f2 = self.banner_label.font(); f2.setPointSize(36); f2.setBold(True)
        self.banner_label.setFont(f2)
        layout.addWidget(self.banner_label, 1)

        self.instruction_label = QtWidgets.QLabel('')
        self.instruction_label.setAlignment(QtCore.Qt.AlignCenter)
        f3 = self.instruction_label.font(); f3.setPointSize(18)
        self.instruction_label.setFont(f3)
        self.instruction_label.setStyleSheet('color: #555;')
        self.instruction_label.setWordWrap(True)
        layout.addWidget(self.instruction_label, 1)

        self.resize(900, 700)

    def update(self):
        self.stimulus_label.setText(_current_stimulus)
        self.instruction_label.setText(_current_instruction)
        if _rec_active:
            self.banner_label.setText('WRITING START!')
            self.banner_label.setStyleSheet('color: #2ca02c;')
        else:
            self.banner_label.setText('writing end — please wait')
            self.banner_label.setStyleSheet('color: #888;')


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    global MIC_TARGET_CH, MIC_GAIN, _mic_target_ch
    global disp_surface_wave, disp_surface_spec, disp_watch_wave, disp_watch_spec
    global _mic_sr_runtime, _trial_dataset_root, _trial_margin, _audio_trial_margin
    global _touch_median_buf, _touch_min_on_sec, _touch_min_off_sec
    global _label_getter, _display_finger, _verbose

    parser = argparse.ArgumentParser(description='WristPad experiment collector')
    parser.add_argument('--mic-device', type=int, default=None)
    parser.add_argument('--mic-channel', type=int, default=1)
    parser.add_argument('--mic-sr', type=int, default=MIC_SR)
    parser.add_argument('--mic-gain', type=float, default=1.0)
    parser.add_argument('--watch-port', type=int, default=WATCH_PORT)
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--camera-pitch-deg', type=float, default=None)
    parser.add_argument('--camera-roll-deg', type=float, default=0.0)
    parser.add_argument('--no-camera', action='store_true')
    parser.add_argument('--finger', choices=FINGER_NAMES, default='index')
    parser.add_argument('--dataset-root', type=Path, default=Path('dataset'))
    parser.add_argument('--session-label', default='')
    parser.add_argument('--trial-margin', type=float, default=0.1)
    parser.add_argument('--audio-trial-margin', type=float, default=0.0)
    parser.add_argument('--material', choices=list(MATERIAL_PRESETS), default='wood')
    parser.add_argument('--touch-min-on-ms', type=float, default=30.0)
    parser.add_argument('--touch-min-off-ms', type=float, default=250.0)
    parser.add_argument('--touch-median-window', type=int, default=3)
    parser.add_argument('--window-sec', type=float, default=2.0)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--display-hz', type=int, default=8000)
    parser.add_argument('--opengl', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--list-devices', action='store_true')
    args = parser.parse_args()
    _verbose = args.verbose

    if args.list_devices:
        print('\n=== Audio Devices ===')
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                print(f'  [{i:2d}] {d["name"]} (in={d["max_input_channels"]}, sr={int(d["default_samplerate"])})')
        return

    app = pg.mkQApp('WristPad Experiment Collector')

    def _handle_sigint(signum, frame):
        print('\n[RUN] Ctrl+C received — shutting down...')
        app.quit()
    signal.signal(signal.SIGINT, _handle_sigint)

    _display_finger = args.finger
    MIC_TARGET_CH = args.mic_channel
    MIC_GAIN = args.mic_gain

    mic_device = args.mic_device
    if mic_device is None:
        for i, d in enumerate(sd.query_devices()):
            name = d['name'].lower()
            if ('tascam' in name or 'us-4x4' in name) and d['max_input_channels'] > 0:
                mic_device = i
                break
            if ('focusrite' in name or 'scarlett' in name) and d['max_input_channels'] > 0:
                mic_device = i
                if MIC_TARGET_CH == 1:
                    MIC_TARGET_CH = 2
                break
    if mic_device is None:
        print('[MIC] No audio interface found. Specify one with --mic-device N.')
        sys.exit(1)
    _mic_target_ch = MIC_TARGET_CH - 1

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    start_session(label=args.session_label)

    _mic_sr_runtime = args.mic_sr
    _trial_dataset_root = args.dataset_root
    _trial_margin = args.trial_margin
    _audio_trial_margin = args.audio_trial_margin
    _trial_dataset_root.mkdir(parents=True, exist_ok=True)

    _touch_median_buf = deque(maxlen=max(1, args.touch_median_window))
    _touch_min_on_sec = args.touch_min_on_ms / 1000.0
    _touch_min_off_sec = args.touch_min_off_ms / 1000.0
    band_low, band_high = MATERIAL_PRESETS[args.material]
    global _current_material
    _current_material = args.material
    _rebuild_touch_band_filter(args.mic_sr, band_low, band_high)

    surface_decimate = max(1, args.mic_sr // args.display_hz)
    disp_surface_wave = ScrollingWaveform(args.mic_sr, args.window_sec, decimate=surface_decimate)
    disp_surface_spec = ScrollingSpectrogram(args.mic_sr, n_fft=1024, hop_sec=0.02,
                                              max_cols=int(args.window_sec / 0.02), freq_max=10000.0)
    watch_decimate = max(1, WATCH_AUDIO_SR // args.display_hz)
    disp_watch_wave = ScrollingWaveform(WATCH_AUDIO_SR, args.window_sec, decimate=watch_decimate)
    disp_watch_spec = ScrollingSpectrogram(WATCH_AUDIO_SR, n_fft=1024, hop_sec=0.03,
                                            max_cols=int(args.window_sec / 0.03), freq_max=6000.0)

    mic_stream = sd.InputStream(
        device=mic_device, channels=MIC_CHANNELS, samplerate=args.mic_sr,
        blocksize=MIC_BLOCK_SIZE, dtype='float32', callback=_mic_callback,
    )
    mic_stream.start()

    net_t = threading.Thread(target=_net_thread_fn, args=(args.watch_port,), daemon=True)
    net_t.start()

    watch_audio_worker_t = threading.Thread(target=_watch_audio_worker_fn, daemon=True)
    watch_audio_worker_t.start()
    watch_imu_worker_t = threading.Thread(target=_watch_imu_worker_fn, daemon=True)
    watch_imu_worker_t.start()

    cam_proc = cam_bridge_t = record_queue = cam_stop_flag = None
    if not args.no_camera:
        record_queue = mp.Queue(maxsize=8)
        cam_stop_flag = mp.Event()
        cam_proc = mp.Process(
            target=_camera_process_fn,
            args=(args.camera_index, args.camera_pitch_deg, args.camera_roll_deg,
                  _session_start, record_queue, cam_stop_flag),
            daemon=True,
        )
        cam_proc.start()
        cam_bridge_t = threading.Thread(target=_camera_bridge_thread_fn, args=(record_queue,), daemon=True)
        cam_bridge_t.start()

    trial_worker_t = threading.Thread(target=_trial_worker_fn, daemon=True)
    trial_worker_t.start()

    audio_worker_t = threading.Thread(target=_audio_worker_fn, daemon=True)
    audio_worker_t.start()

    mic_wav_writer_t = threading.Thread(target=_mic_wav_writer_fn, daemon=True)
    mic_wav_writer_t.start()

    key_listener = keyboard.Listener(on_press=_on_key_press, on_release=_on_key_release)
    try:
        key_listener.start()
    except Exception:
        pass

    instructor = InstructorWindow(args.window_sec, has_camera=not args.no_camera,
                                   use_opengl=args.opengl)
    instructor.label_edit.setText(args.session_label)
    _label_getter = lambda: instructor.label_edit.text()

    experimenter = ExperimenterWindow()

    screens = QtGui.QGuiApplication.screens()
    if len(screens) >= 2:
        instructor.setGeometry(screens[0].geometry())
        experimenter.setGeometry(screens[1].geometry())
        experimenter.showMaximized()
    else:
        experimenter.move(instructor.x() + 50, instructor.y() + 50)
    instructor.show()
    experimenter.show()

    timer = QtCore.QTimer()
    def _tick():
        instructor.update()
        experimenter.update()
    timer.timeout.connect(_tick)
    timer.start(max(1, int(1000 / args.fps)))

    _shutdown_done = False

    def _shutdown():
        nonlocal _shutdown_done
        if _shutdown_done:
            return
        _shutdown_done = True
        stop_event.set()
        if cam_stop_flag:
            cam_stop_flag.set()
        mic_stream.stop(); mic_stream.close()
        mic_wav_writer_t.join(timeout=2.0)
        net_t.join(timeout=2.0)
        watch_audio_worker_t.join(timeout=2.0)
        watch_imu_worker_t.join(timeout=2.0)
        if cam_proc:
            cam_proc.join(timeout=2.0)
            if cam_proc.is_alive():
                cam_proc.terminate()
        if cam_bridge_t:
            cam_bridge_t.join(timeout=2.0)
        try:
            key_listener.stop()
        except Exception:
            pass
        trial_worker_t.join(timeout=2.0)
        audio_worker_t.join(timeout=2.0)
        close_session()

    app.aboutToQuit.connect(_shutdown)

    try:
        if hasattr(pg, 'exec'):
            pg.exec()
        else:
            app.exec_()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown()


if __name__ == '__main__':
    main()