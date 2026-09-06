"""
unified_collector.py
────────────────────────────────────────────────────────────────────────────
Galaxy Watch 7 (TCP/WiFi) + VS-BV203-B Surface Mic (TASCAM USB)
+ MediaPipe-based 5-finger fingertip virtual IMU unified collector

Collected data streams:
  1) Watch audio        48kHz  → watch_audio.wav
  2) Watch IMU acc/gyro         → imu.csv
  3) Surface Mic        192kHz → surface_mic.wav
  4) Fingertip virtual IMU (5 fingers, accel/gyro) → fingertip_imu.csv
  5) Index-finger 2D/3D/global trajectory          → trajectory.csv
  6) Spacebar touch-down/up marking            → events.csv
  7) Sync info                  → sync.json

Trajectory calibration (see trajectory_calibration.py):
  Runs once, interactively, before each session starts (before the camera
  subprocess opens the device) — reuses a saved calibration.json if present,
  press 'c' to (re)calibrate, any other key to start recording. Pass
  --skip-calibration to bypass the prompt entirely (silently reuses
  calibration.json if present, else records uncalibrated trajectory values).

Sync:
  Common reference clock via time.perf_counter() (shared across all streams,
  including the fingertip IMU and trajectory).
  The first-received timestamp of each stream is recorded as an offset (in
  seconds) relative to session_start.

Storage layout:
  data/session_YYYYMMDD_HHMMSS[_label]/
  ├── watch_audio.wav      (48kHz, mono, int16)
  ├── surface_mic.wav      (192kHz, mono, int16)
  ├── imu.csv              (timestamp_sec, sensor, v1, v2, v3, watch_ts_ms)
  ├── fingertip_imu.csv    (timestamp_sec, finger, hand_label, detected,
  │                         accel_x/y/z, gyro_x/y/z, pos_x/y/z)
  ├── trajectory.csv       (timestamp_sec, detected, x_px, y_px, x_norm, y_norm,
  │                         pos_x/y/z_mm, local_x/y/z_mm, calibrated,
  │                         global_x/y_mm, height_mm, frame_x/y_mm)
  ├── events.csv           (timestamp_sec, event: touch_down/touch_up)
  └── sync.json

  dataset/<label>/trial_NNN/ additionally gets a per-trial trajectory.csv
  (same columns, time_aligned to the trial's own start) alongside the
  existing imu.csv/fingertip_imu.csv/watch_audio.wav/surface_mic.wav.

Usage:
    python unified_collector.py
    python unified_collector.py --label exp01
    python unified_collector.py --mic-device 1 --mic-channel 1
    python unified_collector.py --list-devices
    python unified_collector.py --camera-index 0 --show-camera      # enable camera preview window
    python unified_collector.py --no-camera                         # disable camera (fingertip IMU + trajectory)
    python unified_collector.py --skip-calibration                  # bypass the interactive calibration prompt
    python unified_collector.py --camera-pitch-deg 30                # 30-degree downward camera tilt → gravity compensation
    python unified_collector.py --label line_horizontal --dataset-root dataset/   # set label/path for live trial saving
    python unified_collector.py --label line_horizontal --trial-margin 0.15        # adjust margin
"""

from __future__ import annotations

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

import cv2
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from pynput import keyboard
from scipy.signal import butter, sosfilt, sosfilt_zi, resample_poly

from trajectory_calibration import load_calibration, trajectory_csv_row, TRAJECTORY_CSV_HEADER

# Note: MultiFingertipIMUTracker / gravity_vector_from_camera_tilt /
# compute_trajectory / run_calibration are imported inside
# _camera_process_fn / _run_precalibration_process instead of here, since
# that code now runs in a separate process (see the comments above each
# function).

# ─── Config ───────────────────────────────────────────────────────────────────
WATCH_HOST       = '0.0.0.0'
WATCH_PORT       = 50005
WATCH_AUDIO_SR   = 48000
WATCH_FRAME_SIZE = WATCH_AUDIO_SR // 25   # 1920 samples
WATCH_BUF_SIZE   = WATCH_FRAME_SIZE * 2   # 3840 bytes

# The watch only timestamps an audio frame once recorder.read() returns,
# which happens after the full ~40ms frame has already been buffered by the
# hardware — so every watch-audio frame's watch_ts_ms is systematically late
# relative to when its audio was actually captured. This can't be fixed on
# the watch side by shifting individual frame timestamps (that cancels out
# against RTBGN, which is timestamped the same way — see the conversation
# that worked this out). Instead it's corrected here, once, after the
# RTBGN-based watch-clock → PC-time mapping. Empirically measured at
# ~30-70ms across sessions (via plot_audio_sync.py's onset comparison against
# surface_mic, which has no such buffering delay). Only applies to watch
# audio — IMU samples are each timestamped individually on the watch side and
# don't have this frame-buffering artifact.
WATCH_AUDIO_LATENCY_SEC = 0.045

MIC_SR           = 192000
MIC_CHANNELS     = 4
MIC_TARGET_CH    = 1      # 1-indexed
MIC_BLOCK_SIZE   = 512
MIC_GAIN         = 1.0

# Fingertip virtual IMU (camera) settings
CAM_SMOOTHING_WINDOW = 3
CAM_EMA_ALPHA        = 0.2
CAM_FLUSH_EVERY_N    = 10   # flush every N frames (records for 5 fingers)

# Index-finger trajectory trail overlay (--show-camera preview only)
TRAJ_TRAIL_MAXLEN = 60
TRAJ_TRAIL_COLOR   = (60, 220, 255)   # BGR

DATA_ROOT        = Path('data')
SESSION_PREFIX   = 'session'

# ─── Global state ─────────────────────────────────────────────────────────────
_lock = threading.Lock()

_session_dir:   Path | None = None
_session_start: float | None = None
# time.time() reference for the same instant as _session_start. perf_counter()
# is only guaranteed monotonic *within* a process — Python's docs don't promise
# a shared epoch across processes, and empirically it isn't shared here. The
# camera subprocess timestamps its frames off this wall-clock reference instead
# so its timestamps land on the same timeline as _offset() in this process.
_session_start_wall: float | None = None

_watch_wf:   wave.Wave_write | None = None
_mic_wf:     wave.Wave_write | None = None
_imu_fp      = None
_imu_writer  = None
_imu_flush_n = 0

_cam_fp      = None   # fingertip_imu.csv file handle
_cam_writer  = None
_cam_flush_n = 0
# Note: the camera + MediaPipe tracker no longer live in this process at all
# (see _camera_process_fn) — there's nothing to hold a reference to here.

_traj_fp      = None   # trajectory.csv file handle (index-finger 2D/3D/global trajectory)
_traj_writer  = None
_traj_flush_n = 0

# Calibration for trajectory tracking (see trajectory_calibration.py) — loaded
# or interactively (re)established once before recording starts (main()),
# then passed as a plain dict into _camera_process_fn since that process gets
# its own fresh module state under 'spawn' and can't read this global.
_calibration: dict | None = None

_events_fp     = None   # events.csv file handle for spacebar down/up marking
_events_writer = None
_space_down    = False  # guards against key auto-repeat firing multiple events

# Per-frame watch-audio index (session-level), used only for post-session
# recalibration once RTEND is known — see _recalibrate_session_trials().
# Records where each frame landed in watch_audio.wav (by sample offset) and
# its own watch_ts_ms, so trials can be re-cropped later with a corrected
# two-point (RTBGN+RTEND) time mapping instead of the single-point mapping
# available live.
_watch_audio_frames_fp:     None = None
_watch_audio_frames_writer: None = None
_watch_audio_session_samples: int = 0

# ─── Live trial buffering ──────────────────────────────────────────────────────
# While touch_down~touch_up is active, each stream's samples are also buffered
# in memory (in addition to being written to the session-level files), so that
# on touch_up the trial can be cropped immediately and handed off to live
# processing (classification/inference).
_trial_lock       = threading.Lock()
_trial_active     = False
_trial_start_offset: float | None = None
_trial_buffers = {
    # imu, watch_audio, fingertip, and trajectory all arrive with real
    # processing latency (network batches for the watch streams; an extra
    # camera-subprocess -> queue -> bridge-thread hop for fingertip/
    # trajectory), so all four are handled via rolling buffers instead
    # (_imu_rolling, _watch_audio_rolling, _fingertip_rolling,
    # _trajectory_rolling; see below) snapshotted post-grace-period in
    # _trial_worker_fn, not gated live by touch state here. Only mic — which
    # arrives directly in this process via a real-time audio callback, with
    # no such lag — is gated directly by the touch state and buffered here.
    'mic': [],   # (ts, np.ndarray float32)
}
_trial_queue: "queue.Queue" = queue.Queue()   # carries (start_offset, end_offset, snapshot)
_mic_sr_runtime: int = 192000   # updated in main() to the actual mic sample rate in use

# ── Pending-trial tracking ─────────────────────────────────────────────────────
# If trials are captured faster than _trial_worker_fn can process them (e.g.
# rapid-fire spacebar presses while collecting strokes), trials pile up in
# _trial_queue and each one waits its turn behind the fixed per-trial grace
# sleep. During that wait, ROLLING_RETENTION_SEC alone would keep pruning the
# rolling buffers based on wall-clock age, and could delete samples belonging
# to a trial that is still queued — silently truncating its IMU/watch-audio
# portion while fingertip/mic (which aren't retention-limited) stay full
# length. To prevent this, we track the start offset of every trial that has
# been queued but not yet had its rolling-buffer snapshot taken, and never
# prune past the oldest one of those, regardless of ROLLING_RETENTION_SEC.
_pending_lock:   threading.Lock = threading.Lock()
_pending_starts: list = []

# ── IMU / watch-audio / fingertip / trajectory rolling buffers ─────────────────
# Watch IMU/audio are transmitted in network batches and always arrive with some
# delay; fingertip/trajectory go through an extra camera-subprocess -> queue ->
# bridge-thread hop that can lag behind real time under load (MediaPipe
# inference, --show-camera drawing, etc). Gating any of these purely by
# _trial_active — checked at the moment each item is finally processed, not
# when it was captured — silently drops (or in the fingertip/trajectory case,
# can drop *all* of) a trial's data once that processing lag exceeds the
# trial's own duration, since by the time the item is processed _trial_active
# has already flipped back to False. Instead, all four streams are buffered
# continuously regardless of touch state, and after touch_up we wait out a
# grace period before extracting the [trial_start, trial_end] window from the
# rolling buffer.
#
# If trials are queued faster than the worker can process them, pruning by
# ROLLING_RETENTION_SEC alone could delete a still-pending trial's data before
# its snapshot is taken. See _pending_trial_starts / _rolling_cutoff() below
# for the mechanism that prevents this.
ROLLING_RETENTION_SEC = 30.0   # hard upper bound on rolling buffer age (safety net against unbounded memory growth).
                                # This alone is NOT what protects trial data from being dropped when the worker
                                # falls behind — see _pending_trial_starts below, which is the real guard.
IMU_GRACE_SEC         = 0.5    # time to wait after touch_up for pending IMU batches to arrive (sec)
WATCH_AUDIO_GRACE_SEC = 0.5    # time to wait after touch_up for pending watch-audio batches to arrive (sec)

# Fingertip/trajectory go through camera-subprocess -> queue -> bridge-thread,
# whose lag is much larger and more variable than network delay (MediaPipe
# inference + --show-camera drawing under load can put it seconds behind real
# time) — a single fixed short sleep like IMU/WATCH_AUDIO_GRACE_SEC isn't
# reliable here; it was tried first and still produced empty fingertip/
# trajectory trial data whenever the camera pipeline's actual backlog
# exceeded that fixed wait. Poll until the rolling buffer has genuinely
# caught up to this trial's own end time instead, capped by CAM_MAX_WAIT_SEC
# so a stalled/dead camera process can't hang trial processing forever.
CAM_CATCHUP_POLL_SEC = 0.05
CAM_MAX_WAIT_SEC      = 5.0

_rolling_lock: threading.Lock = threading.Lock()
_imu_rolling:         "deque" = deque()   # (ts, sensor, v1, v2, v3, watch_ts_ms)
_watch_audio_rolling: "deque" = deque()   # (ts, raw_bytes, watch_ts_ms)
_fingertip_rolling:   "deque" = deque()   # FingertipIMURecord objects, as-is
_trajectory_rolling:  "deque" = deque()   # (ts, trajectory dict from compute_trajectory())

# Live trial-saving configuration — updated from CLI args in main()
_trial_label:        str  = ''            # class label being repeated in this session (reuses --label)
_trial_dataset_root: Path = Path('dataset')
_trial_margin:       float = 0.1          # margin trimmed at touch_down/up boundaries to compensate for reaction delay (sec)

_watch_audio_offset: float | None = None
_mic_offset:         float | None = None
_imu_offset:         float | None = None
_cam_offset:         float | None = None   # offset of the first fingertip IMU sample
_sync: dict = {}

# RTBGN/RTEND watch-clock ↔ PC-clock mapping
_rtbgn_watch_ms: float | None = None
_rtbgn_pc_sec:   float | None = None
_rtend_watch_ms: float | None = None
_rtend_pc_sec:   float | None = None

# Live RMS values (for STATUS display)
_watch_rms: float = 0.0
_mic_rms:   float = 0.0

# Surface mic high-pass filter (removes DC below 10 Hz)
_mic_sos = butter(2, 10.0 / (MIC_SR / 2), btype='high', output='sos')
_mic_zi  = sosfilt_zi(_mic_sos) * 0

stop_event = threading.Event()


# ─── Utilities ─────────────────────────────────────────────────────────────────
def _offset() -> float:
    if _session_start is None:
        return 0.0
    return time.perf_counter() - _session_start

def _log(tag: str, msg: str):
    print(f'\n[{_offset():8.3f}s][{tag}] {msg}')


def _rolling_cutoff(now_ts: float) -> float:
    """Returns the timestamp before which rolling-buffer entries may be
    discarded. Normally this is just `now_ts - ROLLING_RETENTION_SEC`, but if
    a trial is still queued/awaiting its rolling-buffer snapshot, the cutoff
    is clamped to that trial's start so its data can't be pruned out from
    under it."""
    age_based_cutoff = now_ts - ROLLING_RETENTION_SEC
    with _pending_lock:
        if _pending_starts:
            return min(age_based_cutoff, min(_pending_starts))
    return age_based_cutoff


# ─── Session management ─────────────────────────────────────────────────────────
def start_session(label: str = '') -> Path:
    global _session_dir, _session_start, _session_start_wall
    global _watch_wf, _mic_wf, _imu_fp, _imu_writer
    global _cam_fp, _cam_writer
    global _traj_fp, _traj_writer
    global _events_fp, _events_writer
    global _watch_audio_frames_fp, _watch_audio_frames_writer, _watch_audio_session_samples
    global _watch_audio_offset, _mic_offset, _imu_offset, _cam_offset, _sync

    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    name   = f'{SESSION_PREFIX}_{ts_str}' + (f'_{label}' if label else '')
    _session_dir   = DATA_ROOT / name
    _session_dir.mkdir(parents=True, exist_ok=True)
    _session_start = time.perf_counter()
    _session_start_wall = time.time()

    _watch_audio_offset = None
    _mic_offset         = None
    _imu_offset         = None
    _cam_offset         = None

    _watch_wf = wave.open(str(_session_dir / 'watch_audio.wav'), 'wb')
    _watch_wf.setnchannels(1);  _watch_wf.setsampwidth(2);  _watch_wf.setframerate(WATCH_AUDIO_SR)

    _mic_wf = wave.open(str(_session_dir / 'surface_mic.wav'), 'wb')
    _mic_wf.setnchannels(1);    _mic_wf.setsampwidth(2);    _mic_wf.setframerate(MIC_SR)

    _imu_fp     = open(_session_dir / 'imu.csv', 'w', newline='')
    _imu_writer = csv.writer(_imu_fp)
    _imu_writer.writerow(['timestamp_sec', 'sensor', 'v1', 'v2', 'v3', 'watch_ts_ms'])

    # Fingertip virtual IMU (camera, 5 fingers)
    _cam_fp     = open(_session_dir / 'fingertip_imu.csv', 'w', newline='')
    _cam_writer = csv.writer(_cam_fp)
    _cam_writer.writerow([
        'timestamp_sec', 'finger', 'hand_label', 'detected',
        'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'pos_x', 'pos_y', 'pos_z',
    ])

    # Index-finger 2D/3D/global trajectory (see trajectory_calibration.py)
    _traj_fp     = open(_session_dir / 'trajectory.csv', 'w', newline='')
    _traj_writer = csv.writer(_traj_fp)
    _traj_writer.writerow(TRAJECTORY_CSV_HEADER)

    # Spacebar touch-down/up marking
    _events_fp     = open(_session_dir / 'events.csv', 'w', newline='')
    _events_writer = csv.writer(_events_fp)
    _events_writer.writerow(['timestamp_sec', 'event'])

    # Per-frame watch-audio index, for post-session recalibration
    _watch_audio_frames_fp     = open(_session_dir / 'watch_audio_frames.csv', 'w', newline='')
    _watch_audio_frames_writer = csv.writer(_watch_audio_frames_fp)
    _watch_audio_frames_writer.writerow(['sample_offset', 'num_samples', 'watch_ts_ms'])
    _watch_audio_session_samples = 0

    _sync = {
        'session_start_epoch':    time.time(),
        'label':                  label,
        'watch_audio_sr':         WATCH_AUDIO_SR,
        'surface_mic_sr':         MIC_SR,
        'watch_audio_offset_sec': None,
        'surface_mic_offset_sec': None,
        'imu_offset_sec':         None,
        'fingertip_imu_offset_sec': None,
        'rtbgn_watch_ms':         None,
        'rtbgn_pc_sec':           None,
        'rtend_watch_ms':         None,
        'rtend_pc_sec':           None,
    }
    print(f'[SESSION] Started → {_session_dir}')
    return _session_dir


def close_session():
    global _watch_wf, _mic_wf, _imu_fp, _cam_fp, _traj_fp, _events_fp, _watch_audio_frames_fp

    for wf in [_watch_wf, _mic_wf]:
        if wf:
            wf.close()
    _watch_wf = _mic_wf = None

    if _imu_fp:
        _imu_fp.close()
        _imu_fp = None

    if _cam_fp:
        _cam_fp.close()
        _cam_fp = None

    if _traj_fp:
        _traj_fp.close()
        _traj_fp = None

    if _events_fp:
        _events_fp.close()
        _events_fp = None

    if _watch_audio_frames_fp:
        _watch_audio_frames_fp.close()
        _watch_audio_frames_fp = None

    if _session_dir:
        with open(_session_dir / 'sync.json', 'w') as f:
            json.dump(_sync, f, indent=2)
        print(f'\n[SESSION] Saved → {_session_dir}')
        for k in ('watch_audio_offset_sec', 'surface_mic_offset_sec', 'imu_offset_sec',
                  'fingertip_imu_offset_sec',
                  'rtbgn_watch_ms', 'rtbgn_pc_sec', 'rtend_watch_ms', 'rtend_pc_sec'):
            print(f'  {k} = {_sync.get(k)}')
        _check_watch_connection_quality()
        _recalibrate_session_trials(_session_dir, _trial_dataset_root)


def _check_watch_connection_quality():
    """Compares elapsed time on the watch's own clock (RTBGN→RTEND) against
    elapsed time on the PC clock over the same interval. A large discrepancy
    means the watch connection stalled or throttled at some point during the
    session (WiFi drop, screen/app going to sleep, etc.), which can leave
    trial-level crops looking misaligned even though this fix pads gaps as
    they're detected. This is a diagnostic warning only — it does not affect
    what gets saved."""
    rtbgn_watch_ms = _sync.get('rtbgn_watch_ms')
    rtbgn_pc_sec   = _sync.get('rtbgn_pc_sec')
    rtend_watch_ms = _sync.get('rtend_watch_ms')
    rtend_pc_sec   = _sync.get('rtend_pc_sec')
    if not (rtbgn_watch_ms and rtbgn_pc_sec and rtend_watch_ms and rtend_pc_sec):
        return

    watch_elapsed = (rtend_watch_ms - rtbgn_watch_ms) / 1000.0
    pc_elapsed    = rtend_pc_sec - rtbgn_pc_sec
    if pc_elapsed <= 0:
        return

    ratio = watch_elapsed / pc_elapsed
    print(f'[QUALITY] watch-clock elapsed={watch_elapsed:.2f}s  PC-clock elapsed={pc_elapsed:.2f}s  ratio={ratio:.2%}')
    if ratio < 0.95:
        print(f'[QUALITY] WARNING: watch connection may have stalled during this session '
              f'(only {ratio:.1%} of real time accounted for on the watch clock). '
              f'Consider re-collecting this session.')
    elif ratio > 1.05:
        print(f'[QUALITY] NOTE: watch clock ran {ratio:.1%} of PC-clock speed this session — '
              f'this is normal clock-rate drift and has already been corrected for by '
              f'the post-session recalibration below.')


def _read_wav_samples(path: Path):
    """Reads a mono 16-bit PCM wav file, returning (sample_rate, int16 ndarray)."""
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate()
        n  = wf.getnframes()
        data = np.frombuffer(wf.readframes(n), dtype='<i2')
    return sr, data


def _recalibrate_session_trials(session_dir: Path, dataset_root: Path):
    """Live trial cropping can only use RTBGN (known at session start) to map
    watch_ts_ms → PC time, which corrects the starting offset but not the
    watch-vs-PC clock RATE (drift accumulates the longer the session runs).
    RTEND is only known once the session ends — so this runs at that point,
    building a corrected two-point linear mapping from RTBGN *and* RTEND, and
    re-crops every trial recorded in this session directly from the
    untouched session-level raw files (watch_audio.wav + its per-frame index,
    and imu.csv), overwriting each trial's watch_audio.wav/imu.csv with the
    corrected version. Surface mic and fingertip IMU aren't affected by watch
    clock drift, so their trial crops are left as-is.
    """
    sync_path = session_dir / 'sync.json'
    if not sync_path.exists():
        return
    with open(sync_path) as f:
        sync = json.load(f)

    rtbgn_watch_ms = sync.get('rtbgn_watch_ms')
    rtbgn_pc_sec   = sync.get('rtbgn_pc_sec')
    rtend_watch_ms = sync.get('rtend_watch_ms')
    rtend_pc_sec   = sync.get('rtend_pc_sec')
    if not (rtbgn_watch_ms and rtbgn_pc_sec and rtend_watch_ms and rtend_pc_sec):
        _log('RECAL', 'RTBGN/RTEND incomplete — skipping post-session recalibration')
        return

    watch_span_sec = (rtend_watch_ms - rtbgn_watch_ms) / 1000.0
    pc_span_sec    = rtend_pc_sec - rtbgn_pc_sec
    if watch_span_sec <= 0 or pc_span_sec <= 0:
        _log('RECAL', 'invalid RTBGN/RTEND span — skipping recalibration')
        return
    rate = pc_span_sec / watch_span_sec   # PC seconds per watch second

    def aligned_pc(watch_ts_ms: float) -> float:
        return rtbgn_pc_sec + (watch_ts_ms - rtbgn_watch_ms) / 1000.0 * rate

    metadata_path = dataset_root / 'metadata.csv'
    if not metadata_path.exists():
        return
    session_name = session_dir.name
    with open(metadata_path) as f:
        trials = [row for row in csv.DictReader(f) if row['session'] == session_name]
    if not trials:
        _log('RECAL', 'no trials from this session found in metadata.csv — nothing to recalibrate')
        return

    frames_path = session_dir / 'watch_audio_frames.csv'
    wav_path    = session_dir / 'watch_audio.wav'
    imu_path    = session_dir / 'imu.csv'

    frame_rows = []
    wav_samples = None
    if frames_path.exists() and wav_path.exists():
        with open(frames_path) as f:
            frame_rows = list(csv.DictReader(f))
        _, wav_samples = _read_wav_samples(wav_path)
    else:
        _log('RECAL', 'missing watch_audio_frames.csv/watch_audio.wav — skipping audio recalibration')

    imu_rows = []
    if imu_path.exists():
        with open(imu_path) as f:
            imu_rows = list(csv.DictReader(f))
    else:
        _log('RECAL', 'missing imu.csv — skipping IMU recalibration')

    n_done = 0
    for trial in trials:
        label       = trial['label']
        trial_idx   = int(trial['trial_idx'])
        trial_start = float(trial['start_sec'])
        trial_end   = float(trial['end_sec'])
        trial_dir   = dataset_root / label / f'trial_{trial_idx:03d}'
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
                    start_i = int(fr['sample_offset'])
                    n       = int(fr['num_samples'])
                    selected.append((apc, wav_samples[start_i:start_i + n]))
            if selected:
                selected.sort(key=lambda item: item[0])
                corrected = np.concatenate([chunk for _, chunk in selected])
                wavfile.write(trial_dir / 'watch_audio.wav', WATCH_AUDIO_SR, corrected)

        if imu_rows:
            corrected_rows = []
            for row in imu_rows:
                wts_str = row.get('watch_ts_ms', '')
                if not wts_str or wts_str == '0':
                    continue
                apc = aligned_pc(float(wts_str))
                if trial_start <= apc <= trial_end:
                    corrected_rows.append((apc - trial_start, row['sensor'], row['v1'], row['v2'], row['v3']))
            if corrected_rows:
                corrected_rows.sort(key=lambda r: r[0])
                with open(trial_dir / 'imu.csv', 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['time_aligned', 'sensor', 'v1', 'v2', 'v3'])
                    for ta, sensor, v1, v2, v3 in corrected_rows:
                        w.writerow([f'{ta:.6f}', sensor, v1, v2, v3])

        n_done += 1

    _log('RECAL', f'recalibrated {n_done}/{len(trials)} trial(s) using two-point RTBGN/RTEND mapping '
                  f'(rate={rate:.6f} PC-sec per watch-sec, i.e. {(rate-1)*100:+.2f}% drift)')


# ─── Watch audio writer ─────────────────────────────────────────────────────────
def _write_watch_audio(raw_bytes: bytes, watch_ts_ms: float | None = None):
    """watch_ts_ms is this frame's own capture timestamp (System.currentTimeMillis()
    on the watch), sent as an 8-byte prefix per frame. It's stored as-is here,
    exactly like IMU's watch_ts_ms — the RTBGN-based conversion to PC time
    happens later, at trial-crop time in process_trial(), once RTBGN is known
    to have been received. If watch_ts_ms is unavailable (e.g. a legacy
    packet without the timestamp prefix), it's left as None and trial
    cropping falls back to this frame's PC arrival time."""
    global _watch_audio_offset, _watch_rms, _watch_audio_session_samples
    ts = _offset()   # PC arrival time — used only for the session-level offset display and as a fallback
    samples    = np.frombuffer(raw_bytes, dtype='<i2').astype(np.float32)
    _watch_rms = float(np.sqrt(np.mean(samples ** 2)))
    with _lock:
        if _watch_audio_offset is None:
            _watch_audio_offset = ts
            _sync['watch_audio_offset_sec'] = _watch_audio_offset
            _log('WATCH', f'first packet  offset={_watch_audio_offset:.4f}s')
    if _watch_wf:
        _watch_wf.writeframes(raw_bytes)

    n_samples = len(raw_bytes) // 2
    if _watch_audio_frames_writer:
        _watch_audio_frames_writer.writerow([
            _watch_audio_session_samples, n_samples,
            f'{watch_ts_ms:.0f}' if watch_ts_ms else '',
        ])
    _watch_audio_session_samples += n_samples

    with _rolling_lock:   # rolling buffer is filled regardless of touch state
        _watch_audio_rolling.append((ts, raw_bytes, watch_ts_ms))
        cutoff = _rolling_cutoff(ts)
        while _watch_audio_rolling and _watch_audio_rolling[0][0] < cutoff:
            _watch_audio_rolling.popleft()


# ─── IMU writer ─────────────────────────────────────────────────────────────────
def _write_imu(sensor: str, v1: float, v2: float, v3: float, watch_ts_ms: float = 0.0):
    global _imu_offset, _imu_flush_n
    ts = _offset()
    with _lock:
        if _imu_offset is None:
            _imu_offset = ts
            _sync['imu_offset_sec'] = _imu_offset
            _log('IMU', f'first sample  offset={_imu_offset:.4f}s')
        if _imu_writer:
            _imu_writer.writerow([
                f'{ts:.6f}', sensor,
                f'{v1:.6f}', f'{v2:.6f}', f'{v3:.6f}',
                f'{watch_ts_ms:.0f}',
            ])
            _imu_flush_n += 1
            if _imu_flush_n % 40 == 0 and _imu_fp:
                _imu_fp.flush()

    with _rolling_lock:   # rolling buffer is filled regardless of touch state
        _imu_rolling.append((ts, sensor, v1, v2, v3, watch_ts_ms))
        cutoff = _rolling_cutoff(ts)
        while _imu_rolling and _imu_rolling[0][0] < cutoff:
            _imu_rolling.popleft()


# ─── Fingertip virtual IMU writer ───────────────────────────────────────────────
def _write_fingertip_imu(records: list):
    """records: list[FingertipIMURecord] (5 fingers, returned by MultiFingertipIMUTracker.update())"""
    global _cam_offset, _cam_flush_n
    with _lock:
        if _cam_offset is None:
            _cam_offset = _offset()
            _sync['fingertip_imu_offset_sec'] = _cam_offset
            _log('CAM', f'first sample  offset={_cam_offset:.4f}s')
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

    with _rolling_lock:   # rolling buffer filled regardless of touch state — see
                          # comment above _imu_rolling for why this can't be
                          # gated live by _trial_active
        for r in records:
            _fingertip_rolling.append(r)
        cutoff = _rolling_cutoff(_offset())
        while _fingertip_rolling and _fingertip_rolling[0].timestamp < cutoff:
            _fingertip_rolling.popleft()


# ─── Index-finger trajectory writer ─────────────────────────────────────────────
def _write_trajectory(traj: dict):
    """traj: dict returned by trajectory_calibration.compute_trajectory() for
    one frame. Uses the index record's own timestamp (same per-frame value
    already shared by every FingertipIMURecord in that frame's `records`)."""
    global _traj_flush_n
    ts = traj['index_record'].timestamp
    with _lock:
        if _traj_writer:
            _traj_writer.writerow(trajectory_csv_row(ts, traj))
            _traj_flush_n += 1
            if _traj_flush_n % CAM_FLUSH_EVERY_N == 0 and _traj_fp:
                _traj_fp.flush()

    with _rolling_lock:   # rolling buffer filled regardless of touch state — see
                          # comment above _imu_rolling for why this can't be
                          # gated live by _trial_active
        _trajectory_rolling.append((ts, traj))
        cutoff = _rolling_cutoff(_offset())
        while _trajectory_rolling and _trajectory_rolling[0][0] < cutoff:
            _trajectory_rolling.popleft()


# ─── Spacebar touch-down/up marking ─────────────────────────────────────────────
#
# The watch sends a fixed 3-second stream at acquisition start and has no way
# to signal touch-down/up on its own, so we treat the period while the
# spacebar is held down (with the opposite hand) as "currently writing."
# This uses the same _offset() clock as the other streams, so it aligns
# directly with imu.csv and fingertip_imu.csv.
#
# Note: human key-press reaction delay (roughly 100-300 ms) means this is not
# a precise ground-truth timing signal. It should not be used as GT for
# trajectory MAE; it is only meant for roughly cropping trial boundaries for
# primitive/letter classification.

def _write_event(event: str):
    ts = _offset()
    with _lock:
        if _events_writer:
            _events_writer.writerow([f'{ts:.6f}', event])
            _events_fp.flush()   # events are rare, so flush immediately
    _log('EVENT', f'{event}  offset={ts:.4f}s')


def _on_key_press(key):
    global _space_down, _trial_active, _trial_start_offset
    if key == keyboard.Key.space and not _space_down:
        _space_down = True   # guards against auto-repeat firing multiple events
        _write_event('touch_down')
        with _trial_lock:   # start live trial buffering
            for k in _trial_buffers:
                _trial_buffers[k].clear()
            _trial_start_offset = _offset()
            _trial_active = True


def _on_key_release(key):
    global _space_down, _trial_active
    if key == keyboard.Key.space:
        _space_down = False
        with _trial_lock:   # end live trial buffering, snapshot it, and hand off to the worker
            _trial_active = False
            start = _trial_start_offset
            end   = _offset()
            snapshot = {k: list(v) for k, v in _trial_buffers.items()}
        _write_event('touch_up')
        if start is not None:
            with _pending_lock:
                _pending_starts.append(start)
            _trial_queue.put((start, end, snapshot))
    elif key == keyboard.Key.esc:
        stop_event.set()
        return False   # stop the listener


def _wait_for_camera_catchup(target_ts: float):
    """Polls until the fingertip/trajectory rolling buffers' most recent
    entry is at or past target_ts (meaning the camera pipeline's backlog has
    caught up to at least this trial's own end), or gives up after
    CAM_MAX_WAIT_SEC and proceeds with whatever's available — logging a
    warning, since that means this trial's fingertip/trajectory data may
    still come out incomplete."""
    deadline = time.time() + CAM_MAX_WAIT_SEC
    while time.time() < deadline:
        with _rolling_lock:
            ft_ts   = _fingertip_rolling[-1].timestamp if _fingertip_rolling else None
            traj_ts = _trajectory_rolling[-1][0] if _trajectory_rolling else None
        if ft_ts is not None and ft_ts >= target_ts and traj_ts is not None and traj_ts >= target_ts:
            return
        time.sleep(CAM_CATCHUP_POLL_SEC)
    _log('TRIAL', f'camera pipeline had not caught up to this trial\'s end after '
                  f'{CAM_MAX_WAIT_SEC}s — fingertip/trajectory data for it may be incomplete')


# ─── Live trial processing worker ───────────────────────────────────────────────
#
# For each (start, end, snapshot) delivered via the queue at touch_up:
#   0) Watch IMU/audio arrive with batch transmission delay, so we wait out a
#      short fixed grace period for those. Fingertip/trajectory arrive via an
#      extra camera-subprocess -> queue -> bridge-thread hop whose lag is
#      larger and more variable (MediaPipe inference + --show-camera drawing
#      under load), so instead of a fixed sleep we poll until that pipeline
#      has actually caught up to this trial's own end (see
#      _wait_for_camera_catchup) before reading the [start, end] window from
#      the rolling buffers (_imu_rolling, _watch_audio_rolling,
#      _fingertip_rolling, _trajectory_rolling).
#   1) Re-normalize each stream's timestamps relative to the trial start
#      (starting from 0).
#   2) Resample the surface mic to the watch audio sample rate.
#   3) Feed the result directly into a live classification/inference function
#      (see the marked insertion point inside process_trial).

def _trial_worker_fn():
    while not stop_event.is_set() or not _trial_queue.empty():
        try:
            start, end, snapshot = _trial_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # Allow time for delayed watch IMU/audio batches to arrive
        time.sleep(max(IMU_GRACE_SEC, WATCH_AUDIO_GRACE_SEC))
        # Wait for the fingertip/trajectory pipeline specifically to catch up
        # to this trial's own end, however long that actually takes
        _wait_for_camera_catchup(end)

        with _rolling_lock:
            snapshot['imu']         = list(_imu_rolling)
            snapshot['watch_audio'] = list(_watch_audio_rolling)
            snapshot['fingertip']   = list(_fingertip_rolling)
            snapshot['trajectory']  = list(_trajectory_rolling)

        with _pending_lock:   # this trial's data is now safely copied out; stop protecting it
            if start in _pending_starts:
                _pending_starts.remove(start)

        try:
            process_trial(start, end, snapshot)
        except Exception as e:
            _log('TRIAL', f'error while processing: {e}')


def process_trial(start: float, end: float, snapshot: dict):
    """
    Crops the buffered snapshot collected during touch_down~touch_up (applying
    the configured margin) and saves it immediately in the same dataset
    layout used by segment_trials.py (dataset/<label>/trial_XXX/). There is no
    need to separately run sync_align.py + segment_trials.py offline — the
    trial folder is complete the moment touch_up occurs.
    """
    trial_start = start + _trial_margin
    trial_end   = end - _trial_margin
    if trial_end <= trial_start:
        _log('TRIAL', f'window too short after applying margin({_trial_margin}s), skipping '
                       f'(raw duration={end-start:.3f}s)')
        return

    # Crop IMU + re-normalize to time_aligned (relative to trial_start = 0s)
    #
    # Note: using the raw PC-arrival timestamp (ts) directly would make all 20
    # samples in a single watch batch appear to arrive nearly simultaneously
    # (within microseconds), erasing the real ~100 Hz sample spacing (~10 ms).
    # Converting the watch's own clock (watch_ts_ms) via the RTBGN mapping
    # preserves the true intra-batch sample spacing (same principle as the
    # existing offline logic in sync_align.py).
    rtbgn_watch_ms = _sync.get('rtbgn_watch_ms')
    rtbgn_pc_sec   = _sync.get('rtbgn_pc_sec')
    use_rtbgn = bool(rtbgn_watch_ms and rtbgn_pc_sec)

    imu_rel = []
    for (ts, sensor, v1, v2, v3, watch_ts_ms) in snapshot['imu']:
        if use_rtbgn and watch_ts_ms:
            aligned_pc = (watch_ts_ms - rtbgn_watch_ms) / 1000.0 + rtbgn_pc_sec
        else:
            aligned_pc = ts   # fall back to PC-arrival time if RTBGN/watch_ts_ms is unavailable
        if trial_start <= aligned_pc <= trial_end:
            imu_rel.append((aligned_pc - trial_start, sensor, v1, v2, v3))

    # Crop fingertip data + re-normalize to time_aligned
    ft_rel = [
        (r.timestamp - trial_start, r.finger, r.hand_label, int(r.detected),
         r.accel_x, r.accel_y, r.accel_z, r.gyro_x, r.gyro_y, r.gyro_z,
         r.pos_x, r.pos_y, r.pos_z)
        for r in snapshot['fingertip']
        if trial_start <= r.timestamp <= trial_end
    ]

    # Crop index-finger trajectory + re-normalize to time_aligned. Comes from
    # the same camera/PC-clock source as fingertip data above, so it doesn't
    # need RTBGN watch-clock alignment either.
    traj_rel = [
        (ts - trial_start, traj)
        for (ts, traj) in snapshot['trajectory']
        if trial_start <= ts <= trial_end
    ]

    if not imu_rel and not ft_rel:
        _log('TRIAL', 'no valid samples remain after applying margin, skipping')
        return

    # Watch audio: align each frame via the same RTBGN mapping as IMU (using
    # its own per-frame watch_ts_ms), not by PC arrival order. Each frame is
    # sent over its own TCP connection, so arrival order can differ from
    # capture order under network jitter — sorting by aligned time corrects
    # for that, the same way it wouldn't be needed if frames shared one
    # ordered connection.
    wa_frames = []
    for (ts, chunk, watch_ts_ms) in snapshot['watch_audio']:
        if use_rtbgn and watch_ts_ms:
            aligned_pc = (watch_ts_ms - rtbgn_watch_ms) / 1000.0 + rtbgn_pc_sec - WATCH_AUDIO_LATENCY_SEC
        else:
            aligned_pc = ts - WATCH_AUDIO_LATENCY_SEC   # fall back to PC-arrival time if RTBGN/watch_ts_ms is unavailable
        if trial_start <= aligned_pc <= trial_end:
            wa_frames.append((aligned_pc, chunk))
    wa_frames.sort(key=lambda item: item[0])
    wa_bytes = b''.join(chunk for _, chunk in wa_frames)
    wa_samples = np.frombuffer(wa_bytes, dtype='<i2') if wa_bytes else np.array([], dtype=np.int16)

    # Surface mic: concatenate blocks within the margin range and resample to the watch audio sample rate
    mic_blocks = [blk for (ts, blk) in snapshot['mic'] if trial_start <= ts <= trial_end]
    if mic_blocks:
        mic_concat = np.concatenate(mic_blocks).astype(np.float64)
        if _mic_sr_runtime != WATCH_AUDIO_SR:
            g    = gcd(WATCH_AUDIO_SR, _mic_sr_runtime)
            up   = WATCH_AUDIO_SR // g
            down = _mic_sr_runtime // g
            mic_rs = resample_poly(mic_concat, up, down)
        else:
            mic_rs = mic_concat
    else:
        mic_rs = np.array([], dtype=np.float64)
    mic_int16 = np.clip(mic_rs * 32767, -32768, 32767).astype(np.int16)

    # ── Determine trial folder (same incremental numbering as segment_trials.py) ──
    label     = _trial_label if _trial_label else 'unlabeled'
    label_dir = _trial_dataset_root / label
    label_dir.mkdir(parents=True, exist_ok=True)
    trial_idx = len(sorted(label_dir.glob('trial_*'))) + 1
    trial_dir = label_dir / f'trial_{trial_idx:03d}'
    trial_dir.mkdir(parents=True, exist_ok=True)

    # ── Save ──
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

    with open(trial_dir / 'trajectory.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(TRAJECTORY_CSV_HEADER)
        for ta, traj in sorted(traj_rel, key=lambda row: row[0]):
            w.writerow(trajectory_csv_row(ta, traj))

    # Append to metadata.csv (write header if it doesn't exist yet)
    metadata_path = _trial_dataset_root / 'metadata.csv'
    write_header = not metadata_path.exists()
    with open(metadata_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['session', 'label', 'trial_idx', 'start_sec', 'end_sec',
                        'duration_sec', 'margin_sec'])
        w.writerow([
            _session_dir.name if _session_dir else '',
            label, trial_idx,
            f'{trial_start:.6f}', f'{trial_end:.6f}',
            f'{trial_end - trial_start:.6f}', _trial_margin,
        ])

    _log(
        'TRIAL',
        f'saved → {trial_dir}  duration={trial_end-trial_start:.3f}s  '
        f'imu={len(imu_rel)}  fingertip={len(ft_rel)}  trajectory={len(traj_rel)}  '
        f'watch_audio={len(wa_samples)}samples  mic={len(mic_int16)}samples',
    )

    # ── A live classification/trajectory-reconstruction model could be hooked in here ──
    # e.g.:
    #   features = extract_features(imu_rel, ft_rel, wa_samples, mic_int16)
    #   predicted_label = model.predict(features)
    #   print(f'[REALTIME] Predicted: {predicted_label}')


# ─── IMU packet parsing ──────────────────────────────────────────────────────────
def _parse_imu_packet(pkt: bytes, sensor: str) -> int:
    try:
        txt = pkt[5:].decode('utf-8', errors='ignore').strip()
    except Exception:
        return 0

    count = 0
    for sample in txt.split('|'):
        parts = sample.strip().split()
        if len(parts) < 3:
            continue
        try:
            v1, v2, v3 = float(parts[0]), float(parts[1]), float(parts[2])
            watch_ts   = float(parts[3]) if len(parts) >= 4 else 0.0
            _write_imu(sensor, v1, v2, v3, watch_ts)
            count += 1
        except ValueError:
            continue
    return count


# ─── Surface mic callback ────────────────────────────────────────────────────────
def _mic_callback(indata, frames, time_info, status):
    global _mic_offset, _mic_zi, _mic_rms

    if _session_dir is None:
        return
    if status:
        print(f'\n[MIC] {status}')

    ch        = MIC_TARGET_CH - 1
    raw       = indata[:, ch].astype(np.float32)
    filtered, _mic_zi = sosfilt(_mic_sos, raw, zi=_mic_zi)
    amplified = np.clip(filtered * MIC_GAIN, -1.0, 1.0)

    _mic_rms = float(np.sqrt(np.mean(amplified ** 2)))

    with _lock:
        if _mic_offset is None:
            _mic_offset = _offset()
            _sync['surface_mic_offset_sec'] = _mic_offset
            _log('MIC', f'first sample  offset={_mic_offset:.4f}s')

    if _mic_wf:
        _mic_wf.writeframes((amplified * 32767).astype(np.int16).tobytes())

    with _trial_lock:   # live trial buffering
        if _trial_active:
            _trial_buffers['mic'].append((_offset(), amplified.copy()))


# ─── Camera (fingertip IMU) worker process ───────────────────────────────────────
#
# MediaPipe inference is CPU-heavy and, running as an in-process thread,
# competed for the GIL with the audio callback and network thread — under
# load this showed up as the surface mic's captured duration falling short
# of the real trial window. Running the camera + MediaPipe tracker in a
# genuinely separate OS process removes that GIL contention entirely: this
# process's only job is capturing frames and running inference, and it talks
# back to the main process solely through the queues below. It does NOT call
# any of the shared writer functions (_write_fingertip_imu, etc.) directly —
# with multiprocessing's default 'spawn' start method, this process gets its
# own fresh copy of all module-level state, so those shared file handles/
# locks wouldn't actually be shared anyway.
def _camera_process_fn(camera_index: int, show_window: bool,
                        camera_pitch_deg, camera_roll_deg: float,
                        session_start_wall: float,
                        record_queue: "mp.Queue", frame_queue,
                        stop_flag: "mp.Event", calibration: dict | None):
    import cv2 as _cv2
    from fingertip_imu_multi import MultiFingertipIMUTracker, gravity_vector_from_camera_tilt
    from trajectory_calibration import compute_trajectory

    gravity_mm_s2 = None
    if camera_pitch_deg is not None:
        gravity_mm_s2 = gravity_vector_from_camera_tilt(camera_pitch_deg, camera_roll_deg)

    tracker = MultiFingertipIMUTracker(
        max_num_hands=1,
        smoothing_window=CAM_SMOOTHING_WINDOW,
        gravity_mm_s2=gravity_mm_s2,
        ema_alpha=CAM_EMA_ALPHA,
    )
    cap = _cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        record_queue.put(None)   # tells the main process the camera failed to open
        return

    trail = deque(maxlen=TRAJ_TRAIL_MAXLEN)   # index-finger 2D trail, preview only

    while not stop_flag.is_set():
        success, frame = cap.read()
        if not success:
            continue
        frame = _cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        # time.time() (wall clock, seconds since epoch) means the same thing
        # in every process on the machine, unlike perf_counter() — whose docs
        # only guarantee monotonicity *within* a process, not a shared epoch
        # across processes. Subtracting the wall-clock reference captured in
        # the main process keeps this timestamp on the same timeline as
        # _offset() there.
        ts = time.time() - session_start_wall
        records = tracker.update(frame, timestamp=ts)
        traj = compute_trajectory(tracker, records, w, h, calibration)
        try:
            record_queue.put_nowait((records, traj))
        except Exception:
            pass   # main process fell behind — drop this frame's records rather than block capture

        if show_window and frame_queue is not None:
            tracker.draw(frame)
            tracker.draw_axes(frame)

            if traj['x_px'] is not None:
                trail.append((traj['x_px'], traj['y_px']))
            # Fading trail: older points drawn thinner and dimmer (same style
            # as index_trajectory_viewer.py's standalone preview).
            n = len(trail)
            for i in range(1, n):
                alpha = i / n
                thickness = max(1, int(alpha * 4))
                color = tuple(int(c * alpha) for c in TRAJ_TRAIL_COLOR)
                _cv2.line(frame, trail[i - 1], trail[i], color, thickness)
            if trail:
                _cv2.circle(frame, trail[-1], 6, TRAJ_TRAIL_COLOR, -1)

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except Exception:
                    pass
            try:
                frame_queue.put_nowait(frame)
            except Exception:
                pass

    cap.release()
    tracker.close()


def _camera_bridge_thread_fn(record_queue: "mp.Queue"):
    """Pulls (fingertip IMU records, trajectory dict) pairs off the queue fed
    by the camera process and writes them exactly as _camera_thread_fn used
    to. This thread does no CPU-heavy work of its own — the MediaPipe
    inference already happened in the child process — so it doesn't
    meaningfully compete for the GIL with the audio/network threads the way
    the old in-process camera thread did."""
    while not stop_event.is_set():
        try:
            payload = record_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if payload is None:
            _log('CAM', 'camera process reported it could not open the camera')
            continue
        records, traj = payload
        _write_fingertip_imu(records)
        _write_trajectory(traj)


# ─── TCP net thread ───────────────────────────────────────────────────────────────
def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
    """Reads exactly n bytes from conn, tolerating recv() timeouts by simply
    retrying (so a slow trickle of bytes doesn't lose partially-read data the
    way restarting the read from scratch would). Returns None if the
    connection was closed before n bytes arrived, or if shutdown was
    requested mid-read."""
    buf = bytearray()
    while len(buf) < n:
        if stop_event.is_set():
            return None
        try:
            chunk = conn.recv(n - len(buf))
        except socket.timeout:
            continue
        if not chunk:
            return None   # connection closed
        buf.extend(chunk)
    return bytes(buf)


def _dispatch_watch_packet(pkt: bytes):
    """Same message dispatch logic as before the persistent-connection
    change — now called once per length-prefixed frame read off the
    connection, instead of once per one-shot TCP connection."""
    total = len(pkt)

    # New protocol: every audio frame is [8-byte big-endian watch_ts_ms]
    # + [WATCH_BUF_SIZE bytes of PCM audio]. This includes frame 0, which
    # used to be dropped (only its RTBGN header was sent).
    if total == WATCH_BUF_SIZE + 8:
        watch_ts_ms = int.from_bytes(pkt[:8], byteorder='big', signed=False)
        _write_watch_audio(pkt[8:], watch_ts_ms)
        return

    # Legacy fallback: a bare audio frame with no timestamp prefix at all
    # (older watch-side builds). Falls back to PC arrival time for
    # trial-crop alignment.
    if total == WATCH_BUF_SIZE:
        _write_watch_audio(pkt)
        return

    try:
        hdr = pkt[:5].decode('utf-8', errors='ignore')
    except Exception:
        return

    if hdr == 'SUBID':
        _log('NET', f'SUBID  {pkt[6:10].decode("utf-8", errors="ignore").strip()}')

    elif hdr == 'RTBGN':
        pc_sec = _offset()
        if len(pkt) >= 13:
            watch_ms = int.from_bytes(pkt[5:13], byteorder='big', signed=False)
            with _lock:
                _sync['rtbgn_watch_ms'] = watch_ms
                _sync['rtbgn_pc_sec']   = pc_sec
            _log('NET', f'RTBGN  watch_ms={watch_ms}  pc_sec={pc_sec:.4f}s')
        else:
            _log('NET', 'RTBGN (no timestamp — watch-side code needs updating)')

    elif hdr == 'RTEND':
        global _watch_wf
        pc_sec = _offset()
        if len(pkt) >= 13:
            watch_ms = int.from_bytes(pkt[5:13], byteorder='big', signed=False)
            with _lock:
                _sync['rtend_watch_ms'] = watch_ms
                _sync['rtend_pc_sec']   = pc_sec
            _log('NET', f'RTEND  watch_ms={watch_ms}  pc_sec={pc_sec:.4f}s')
        # Stop writing watch audio after RTEND
        if _watch_wf:
            _watch_wf.close()
            _watch_wf = None

    elif hdr == 'SOUND':
        # Legacy non-realtime full-session dump: header + id, no per-frame
        # timestamp. Falls back to PC arrival time for trial-crop alignment.
        raw = pkt[10:]
        buf = np.frombuffer(raw[:len(raw)//2*2], dtype='<i2')
        _write_watch_audio(buf.tobytes())
        _log('NET', f'SOUND  {len(buf)} samples')

    elif hdr == 'IMUAC':
        n = _parse_imu_packet(pkt, 'acc')
        if n:
            _log('NET', f'IMUAC  {n} samples')

    elif hdr == 'IMUGY':
        n = _parse_imu_packet(pkt, 'gyro')
        if n:
            _log('NET', f'IMUGY  {n} samples')

    elif total > 0 and total % 2 == 0:
        _write_watch_audio(pkt)


def _net_thread_fn():
    # Backlog raised from 5 → 16: mostly academic now that the watch keeps a
    # single long-lived connection open (see DataRecorder.java's persistent
    # connection), but it costs nothing and gives more headroom if the watch
    # ever needs to reconnect in a burst.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((WATCH_HOST, WATCH_PORT))
    srv.listen(16)
    srv.settimeout(1.0)

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = 'unknown'
    print(f'[NET] Watch TCP: {local_ip}:{WATCH_PORT}')

    while not stop_event.is_set():
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            continue
        except Exception as e:
            if not stop_event.is_set():
                _log('NET', f'accept error: {e}')
            continue

        _log('NET', f'watch connected from {addr}')
        conn.settimeout(1.0)   # bounds how long a single recv() blocks, so shutdown/reconnect stays responsive
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        try:
            while not stop_event.is_set():
                header = _recv_exact(conn, 4)
                if header is None:
                    break   # watch closed the connection (or shutdown requested)

                msg_len = int.from_bytes(header, byteorder='big', signed=False)
                if msg_len <= 0 or msg_len > 10_000_000:
                    _log('NET', f'implausible message length {msg_len}, dropping connection')
                    break

                payload = _recv_exact(conn, msg_len)
                if payload is None:
                    break

                _dispatch_watch_packet(payload)
        except Exception as e:
            _log('NET', f'connection error: {e}')
        finally:
            conn.close()
            _log('NET', 'watch disconnected — waiting for reconnect')

    srv.close()
    _log('NET', 'stopped')


# ─── Entry point ─────────────────────────────────────────────────────────────────
def _bar(rms: float, scale: float, width: int = 20) -> str:
    filled = int(min(rms / scale, 1.0) * width)
    return '█' * filled + '░' * (width - filled)


def _run_precalibration_process(camera_index: int, result_queue: "mp.Queue"):
    """Runs the interactive trajectory calibration (see trajectory_calibration.py)
    in its own OS process, fully isolated from the main process — a temporary
    VideoCapture + OpenCV GUI window here previously ran in the main process
    itself, which turned out to destabilize it once pynput's keyboard
    listener started up right afterward (both touch low-level macOS
    frameworks in the same process; this showed up as a native 'trace trap'
    crash right at key_listener.start()). Running it here instead means the
    main process never touches OpenCV's GUI at all, and the OS fully
    reclaims the camera device the moment this process exits — before the
    real camera subprocess opens it. Puts the resulting calibration dict (or
    None) on result_queue once done.

    Loads an existing calibration.json if present; press 'c' to
    (re)calibrate, any other key to finish and let main() proceed to
    recording with whatever calibration (if any) is active."""
    import cv2 as _cv2
    from fingertip_imu_multi import MultiFingertipIMUTracker
    from trajectory_calibration import load_calibration as _load_calibration, run_calibration as _run_calibration

    calibration = _load_calibration()
    if calibration is not None:
        print(f'[CALIBRATE] loaded existing calibration from {calibration.get("timestamp", "?")} '
              f'(scale_factor={calibration.get("scale_factor"):.4f})')

    cap = _cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f'[CALIBRATE] could not open camera index={camera_index} for pre-recording calibration '
              f'— continuing with {"loaded" if calibration else "no"} calibration')
        result_queue.put(calibration)
        return

    window_name = 'Trajectory Calibration'
    _cv2.namedWindow(window_name)
    tracker = MultiFingertipIMUTracker(max_num_hands=1)
    print("[CALIBRATE] press 'c' to (re)calibrate trajectory tracking, any other key to start recording")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = _cv2.flip(frame, 1)
            tracker.update(frame, timestamp=time.perf_counter())
            # Draw the live-preview overlay on a copy, not `frame` itself —
            # `frame` (unannotated) is what gets passed into calibration
            # below, so its own on-screen prompts don't end up overlapping
            # this loop's "press c to calibrate" reminder text.
            display = frame.copy()
            tracker.draw(display)
            _cv2.putText(display, "press 'c' to calibrate, any other key to start recording",
                         (10, display.shape[0] - 20), _cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                         (0, 255, 255), 1, _cv2.LINE_AA)
            _cv2.imshow(window_name, display)
            key = _cv2.waitKey(20) & 0xFF
            if key == ord('c'):
                new_calibration = _run_calibration(tracker, frame, window_name)
                if new_calibration is not None:
                    calibration = new_calibration
                continue
            if key != 255:   # any real key press other than 'c' (255 = none this poll)
                break
    finally:
        cap.release()
        _cv2.destroyAllWindows()
        tracker.close()

    result_queue.put(calibration)


def main():
    global MIC_TARGET_CH, MIC_GAIN

    parser = argparse.ArgumentParser(description='WristPad Unified Collector')
    parser.add_argument('--label',        default='')
    parser.add_argument('--mic-device',   type=int,   default=None)
    parser.add_argument('--mic-channel',  type=int,   default=1)
    parser.add_argument('--mic-channels', type=int,   default=MIC_CHANNELS,
                         help=f'total input channel count to open on the mic device '
                              f'(default {MIC_CHANNELS}, sized for a 4-channel interface like the '
                              f'TASCAM US-4x4; set to your device\'s actual channel count, e.g. 2, '
                              f'if PortAudio reports "Invalid number of channels")')
    parser.add_argument('--mic-sr',       type=int,   default=MIC_SR)
    parser.add_argument('--mic-gain',     type=float, default=1.0)
    parser.add_argument('--watch-port',   type=int,   default=WATCH_PORT)
    parser.add_argument('--list-devices', action='store_true')
    # Fingertip virtual IMU (camera) options
    parser.add_argument('--camera-index',     type=int,   default=0)
    parser.add_argument('--no-camera',        action='store_true',
                         help='disable fingertip IMU (camera) collection')
    parser.add_argument('--show-camera',      action='store_true',
                         help='show camera preview window (skeleton + axis overlay)')
    parser.add_argument('--camera-pitch-deg', type=float, default=None,
                         help='downward camera tilt angle; if set, applies gravity compensation')
    parser.add_argument('--camera-roll-deg',  type=float, default=0.0)
    parser.add_argument('--skip-calibration', action='store_true',
                         help="skip the interactive pre-recording trajectory calibration prompt "
                              "(just silently reuses calibration.json if it exists, else records "
                              "uncalibrated trajectory values)")
    # Live trial-saving options
    parser.add_argument('--dataset-root', type=Path, default=Path('dataset'),
                         help='root folder for live trial dataset saving (default: ./dataset)')
    parser.add_argument('--trial-margin', type=float, default=0.1,
                         help='margin trimmed inward from the touch_down/up boundaries (sec), '
                              'to compensate for spacebar reaction delay (default 0.1s)')
    args = parser.parse_args()

    if args.list_devices:
        print('\n=== Audio Devices ===')
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                print(f'  [{i:2d}] {d["name"]} '
                      f'(in={d["max_input_channels"]}, sr={int(d["default_samplerate"])})')
        return

    MIC_TARGET_CH = args.mic_channel
    MIC_GAIN      = args.mic_gain

    mic_device = args.mic_device
    if mic_device is None:
        for i, d in enumerate(sd.query_devices()):
            name = d['name'].lower()
            if ('tascam' in name or 'us-4x4' in name) \
                    and d['max_input_channels'] > 0:
                mic_device = i
                break
            if ('focusrite' in name or 'scarlett' in name) \
                    and d['max_input_channels'] > 0:
                mic_device = i
                # Scarlett Solo: XLR input is on channel 2
                if MIC_TARGET_CH == 1:
                    MIC_TARGET_CH = 2
                    print('[MIC] Scarlett detected → auto-set to channel 2 (XLR input)')
                break
    if mic_device is None:
        print('[MIC] No audio interface found. '
              'Specify one directly with --mic-device N (see --list-devices).')
        sys.exit(1)

    device_max_ch = sd.query_devices(mic_device)['max_input_channels']
    if args.mic_channels > device_max_ch:
        print(f'[MIC] --mic-channels={args.mic_channels} exceeds device [{mic_device}]\'s '
              f'max_input_channels={device_max_ch}. Pass --mic-channels {device_max_ch} '
              f'(see --list-devices for other options).')
        sys.exit(1)
    if MIC_TARGET_CH > args.mic_channels:
        print(f'[MIC] --mic-channel={MIC_TARGET_CH} is out of range for '
              f'--mic-channels={args.mic_channels}.')
        sys.exit(1)
    print(f'[MIC] device=[{mic_device}] channels={args.mic_channels} '
          f'target_ch={MIC_TARGET_CH} sr={args.mic_sr}')

    # Trajectory calibration happens before the session starts (and before
    # the camera subprocess opens the device), so calibration time doesn't
    # count toward the session clock and the two never fight over the camera.
    global _calibration
    if args.no_camera:
        _calibration = None
    elif args.skip_calibration:
        _calibration = load_calibration()
    else:
        calib_result_queue: "mp.Queue" = mp.Queue()
        calib_proc = mp.Process(
            target=_run_precalibration_process,
            args=(args.camera_index, calib_result_queue),
        )
        calib_proc.start()
        calib_proc.join()
        try:
            _calibration = calib_result_queue.get_nowait()
        except queue.Empty:
            print('[CALIBRATE] calibration process exited without a result — '
                  'continuing with any saved calibration')
            _calibration = load_calibration()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    start_session(label=args.label)

    global _mic_sr_runtime   # reference sample rate for live trial resampling
    _mic_sr_runtime = args.mic_sr

    global _trial_label, _trial_dataset_root, _trial_margin
    _trial_label        = args.label
    _trial_dataset_root = args.dataset_root
    _trial_margin        = args.trial_margin
    _trial_dataset_root.mkdir(parents=True, exist_ok=True)
    print(f'[TRIAL] live trial saving → {_trial_dataset_root}/{_trial_label or "unlabeled"}/ '
          f'(margin={_trial_margin}s)')

    mic_stream = sd.InputStream(
        device=mic_device,
        channels=args.mic_channels,
        samplerate=args.mic_sr,
        blocksize=MIC_BLOCK_SIZE,
        dtype='float32',
        callback=_mic_callback,
    )
    mic_stream.start()

    net_t = threading.Thread(target=_net_thread_fn, daemon=True)
    net_t.start()

    # Fingertip virtual IMU: camera capture + MediaPipe inference run in a
    # separate OS process (see _camera_process_fn); a lightweight bridge
    # thread here just relays the results into the normal write path.
    cam_proc = None
    cam_bridge_t = None
    record_queue = None
    frame_queue = None
    cam_stop_flag = None
    if not args.no_camera:
        record_queue  = mp.Queue(maxsize=8)
        frame_queue   = mp.Queue(maxsize=1) if args.show_camera else None
        cam_stop_flag = mp.Event()
        cam_proc = mp.Process(
            target=_camera_process_fn,
            args=(args.camera_index, args.show_camera, args.camera_pitch_deg, args.camera_roll_deg,
                  _session_start_wall, record_queue, frame_queue, cam_stop_flag, _calibration),
            daemon=True,
        )
        cam_proc.start()
        cam_bridge_t = threading.Thread(target=_camera_bridge_thread_fn, args=(record_queue,), daemon=True)
        cam_bridge_t.start()
        _log('CAM', f'started in a separate process (index={args.camera_index}, pid={cam_proc.pid})')
    else:
        print('[CAM] --no-camera specified → fingertip IMU not collected')

    # Spacebar touch-down/up marking listener
    # Records the period while the spacebar is held (with the opposite hand)
    # as the writing interval (ESC to stop early)
    key_listener = keyboard.Listener(on_press=_on_key_press, on_release=_on_key_release)
    key_listener.start()
    print('[EVENT] The period while the spacebar is held is recorded as the writing interval (ESC to stop)')

    # Live trial processing worker thread
    trial_worker_t = threading.Thread(target=_trial_worker_fn, daemon=True)
    trial_worker_t.start()

    def _shutdown(sig=None, frame=None):
        print('\n[RUN] Stopping...')
        stop_event.set()
        if cam_stop_flag:
            cam_stop_flag.set()
        mic_stream.stop();  mic_stream.close()
        net_t.join(timeout=2.0)
        if cam_proc:
            cam_proc.join(timeout=2.0)
            if cam_proc.is_alive():
                cam_proc.terminate()
        if cam_bridge_t:
            cam_bridge_t.join(timeout=2.0)
        key_listener.stop()
        trial_worker_t.join(timeout=2.0)
        if args.show_camera:
            cv2.destroyAllWindows()
        close_session()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print('[RUN] Collecting... Ctrl+C to stop\n')

    def _print_status():
        wa = f'{_watch_audio_offset:.2f}s' if _watch_audio_offset else '–'
        sm = f'{_mic_offset:.2f}s'         if _mic_offset         else '–'
        im = f'{_imu_offset:.2f}s'         if _imu_offset         else '–'
        cm = f'{_cam_offset:.2f}s'         if _cam_offset         else '–'
        wa_rms  = f'{_watch_rms:.0f}'  if _watch_rms  else '–'
        mic_rms = f'{_mic_rms:.4f}'    if _mic_rms    else '–'
        print(
            f'\r[STATUS] Watch:{wa} RMS={wa_rms}  '
            f'Mic:{sm} RMS={mic_rms}  '
            f'IMU:{im}  Cam:{cm}   ',
            end='', flush=True,
        )

    if args.show_camera and frame_queue is not None:
        # On macOS and similar platforms, imshow/waitKey must be called from
        # the main thread, so the main thread pulls frames the camera
        # process placed in frame_queue for display.
        last_status_t = 0.0
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.05)
                cv2.imshow('Fingertip IMU (WristPad)', frame)
            except queue.Empty:
                pass
            if cv2.waitKey(1) & 0xFF == ord('q'):
                _shutdown()
                break
            now = time.time()
            if now - last_status_t >= 1.0:
                _print_status()
                last_status_t = now
    else:
        while True:
            time.sleep(1.0)
            _print_status()


if __name__ == '__main__':
    main()