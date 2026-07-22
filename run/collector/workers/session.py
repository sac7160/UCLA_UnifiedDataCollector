"""
data_collector/workers/session.py
────────────────────────────────────────────────────────────────────────────
Session-level file lifecycle: opening the session_YYYYMMDD_HHMMSS/ folder
and all its files at start_session(), closing them (and writing sync.json)
at close_session(), plus two post-session diagnostics/fixups that only make
sense once the whole session's data is on disk:
  - _check_watch_connection_quality(): compares watch-clock vs. PC-clock
    elapsed time over the session to flag a stalled/throttled connection.
  - _recalibrate_session_trials(): re-crops each trial's watch_audio.wav
    using the two-point RTBGN+RTEND mapping (only available once the
    session has actually ended), correcting for watch-clock drift that the
    single-point RTBGN-only mapping used during live trial saving can't.
"""

import csv
import json
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from ..core import config, state
from .trial import crop_watch_audio_frames
from trajectory_calibration import TRAJECTORY_CSV_HEADER


def start_session(label: str = '') -> Path:
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    name   = f'{config.SESSION_PREFIX}_{ts_str}' + (f'_{label}' if label else '')
    state.session_dir   = config.DATA_ROOT / name
    state.session_dir.mkdir(parents=True, exist_ok=True)
    state.session_start = time.perf_counter()
    state.session_start_wall = time.time()

    state.watch_audio_offset = None
    state.mic_offset         = None
    state.imu_offset         = None
    state.cam_offset         = None
    state.event_log          = []

    state.watch_wf = wave.open(str(state.session_dir / 'watch_audio.wav'), 'wb')
    state.watch_wf.setnchannels(1); state.watch_wf.setsampwidth(2)
    state.watch_wf.setframerate(config.WATCH_AUDIO_SR)

    state.mic_wf = wave.open(str(state.session_dir / 'surface_mic.wav'), 'wb')
    state.mic_wf.setnchannels(1); state.mic_wf.setsampwidth(2)
    state.mic_wf.setframerate(config.MIC_SR)

    state.imu_fp     = open(state.session_dir / 'imu.csv', 'w', newline='')
    state.imu_writer = csv.writer(state.imu_fp)
    state.imu_writer.writerow(['timestamp_sec', 'sensor', 'v1', 'v2', 'v3', 'watch_ts_ms'])

    state.cam_fp     = open(state.session_dir / 'fingertip_imu.csv', 'w', newline='')
    state.cam_writer = csv.writer(state.cam_fp)
    state.cam_writer.writerow([
        'timestamp_sec', 'finger', 'hand_label', 'detected',
        'accel_x', 'accel_y', 'accel_z',
        'gyro_x', 'gyro_y', 'gyro_z',
        'pos_x', 'pos_y', 'pos_z',
    ])

    state.traj_fp     = open(state.session_dir / 'trajectory.csv', 'w', newline='')
    state.traj_writer = csv.writer(state.traj_fp)
    state.traj_writer.writerow(TRAJECTORY_CSV_HEADER)

    state.events_fp     = open(state.session_dir / 'events.csv', 'w', newline='')
    state.events_writer = csv.writer(state.events_fp)
    state.events_writer.writerow(['timestamp_sec', 'event'])

    state.watch_audio_frames_fp     = open(state.session_dir / 'watch_audio_frames.csv', 'w', newline='')
    state.watch_audio_frames_writer = csv.writer(state.watch_audio_frames_fp)
    state.watch_audio_frames_writer.writerow(['sample_offset', 'num_samples', 'watch_ts_ms'])
    state.watch_audio_session_samples = 0
    with state.heartbeat_lock:
        state.heartbeat_audio_frames = 0
        state.heartbeat_imu_acc = 0
        state.heartbeat_imu_gyro = 0

    state.sync = {
        'session_start_epoch':      time.time(),
        'label':                    label,
        'watch_audio_sr':           config.WATCH_AUDIO_SR,
        'surface_mic_sr':           config.MIC_SR,
        'watch_audio_offset_sec':   None,
        'surface_mic_offset_sec':   None,
        'imu_offset_sec':           None,
        'fingertip_imu_offset_sec': None,
        'rtbgn_watch_ms': None, 'rtbgn_pc_sec': None,
        'rtend_watch_ms': None, 'rtend_pc_sec': None,
    }
    print(f'[SESSION] Started -> {state.session_dir}')
    return state.session_dir


def close_session():
    for wf in [state.watch_wf, state.mic_wf]:
        if wf:
            wf.close()
    state.watch_wf = state.mic_wf = None
    if state.imu_fp: state.imu_fp.close(); state.imu_fp = None
    if state.cam_fp: state.cam_fp.close(); state.cam_fp = None
    if state.traj_fp: state.traj_fp.close(); state.traj_fp = None
    if state.events_fp: state.events_fp.close(); state.events_fp = None
    if state.watch_audio_frames_fp:
        state.watch_audio_frames_fp.close()
        state.watch_audio_frames_fp = None

    if state.session_dir:
        with open(state.session_dir / 'sync.json', 'w') as f:
            json.dump(state.sync, f, indent=2)
        print(f'\n[SESSION] Saved -> {state.session_dir}')
        _check_watch_connection_quality()
        _recalibrate_session_trials(state.session_dir, state.trial_dataset_root)


def _check_watch_connection_quality():
    rtbgn_watch_ms = state.sync.get('rtbgn_watch_ms'); rtbgn_pc_sec = state.sync.get('rtbgn_pc_sec')
    rtend_watch_ms = state.sync.get('rtend_watch_ms'); rtend_pc_sec = state.sync.get('rtend_pc_sec')
    if not (rtbgn_watch_ms and rtbgn_pc_sec and rtend_watch_ms and rtend_pc_sec):
        return
    watch_elapsed = (rtend_watch_ms - rtbgn_watch_ms) / 1000.0
    pc_elapsed    = rtend_pc_sec - rtbgn_pc_sec
    if pc_elapsed <= 0:
        return
    ratio = watch_elapsed / pc_elapsed
    print(f'[QUALITY] watch-clock elapsed={watch_elapsed:.2f}s  '
          f'PC-clock elapsed={pc_elapsed:.2f}s  ratio={ratio:.2%}')


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

    frame_rows = []; wav_samples = None
    if frames_path.exists() and wav_path.exists():
        with open(frames_path) as f:
            frame_rows = list(csv.DictReader(f))
        _, wav_samples = _read_wav_samples(wav_path)

    for trial in trials:
        label = trial['label']; trial_idx = int(trial['trial_idx'])
        trial_start = float(trial['start_sec']); trial_end = float(trial['end_sec'])
        trial_dir = dataset_root / label / f'trial_{trial_idx:03d}'
        if not trial_dir.exists():
            continue

        if frame_rows and wav_samples is not None:
            raw_frames = []
            for fr in frame_rows:
                wts_str = fr.get('watch_ts_ms', '')
                if not wts_str:
                    continue
                frame_start_pc = aligned_pc(float(wts_str)) - config.WATCH_AUDIO_LATENCY_SEC
                start_i = int(fr['sample_offset']); n = int(fr['num_samples'])
                raw_frames.append((frame_start_pc, wav_samples[start_i:start_i + n]))
            # Same sample-accurate crop process_trial() uses live — this
            # used to have its own, coarser whole-frame-inclusion version
            # here, which silently re-introduced up to ~40ms of slop at
            # each trial boundary every time a session closed normally and
            # this recalibration overwrote the live-saved (precisely
            # cropped) file. See crop_watch_audio_frames's docstring.
            corrected = crop_watch_audio_frames(raw_frames, trial_start, trial_end, config.WATCH_AUDIO_SR)
            if len(corrected):
                wavfile.write(trial_dir / 'watch_audio.wav', config.WATCH_AUDIO_SR, corrected)