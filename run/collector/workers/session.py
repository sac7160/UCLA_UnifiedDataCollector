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
import shutil
import subprocess
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
        _fix_video_framerate(state.session_dir)
        _extract_trial_videos(state.session_dir, state.trial_dataset_root)


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


def _fix_video_framerate(session_dir: Path):
    """cv2.VideoWriter needs a fixed fps up front, but camera_process_fn's
    real achieved capture rate (cap.read() + MediaPipe inference every
    frame) is neither known in advance nor constant — config.CAM_VIDEO_FALLBACK_FPS
    is only ever a provisional guess (see camera.py, which deliberately
    doesn't even try cap.get(CAP_PROP_FPS): queried before any frame has
    been read, that's commonly wrong too). If the guess is off from the
    true average rate, two things break: played back, camera_raw.avi looks
    sped up or slowed down relative to what actually happened, AND —
    because _extract_trial_videos()'s -ss/-to below is evaluated against
    the container's own internal timestamp track, not real wall-clock
    seconds — per-trial extraction can seek to the wrong content entirely,
    including past the container's own (wrong) idea of where the stream
    ends, silently producing no output for that trial.

    Fixed by relabeling with the true fps measured from camera_frames.csv's
    own per-frame timestamps — the same ground-truth file everything else in
    this pipeline already trusts over container metadata. `ffmpeg -r
    <true_fps> -i ... -c copy` (as an INPUT option, before -i) discards the
    container's existing timestamps and regenerates them at the given
    constant rate; combined with -c copy this is a pure relabel, no
    re-encoding, safe for the same all-intra reason _extract_trial_videos()
    below trusts frame indices to stay meaningful. Independent of it
    though — that function matches frames by their own real timestamps,
    never the container's PTS, so the two can run in either order."""
    video_path = session_dir / f'camera_raw{config.CAM_VIDEO_EXT}'
    frames_path = session_dir / 'camera_frames.csv'
    if not video_path.exists() or not frames_path.exists() or shutil.which('ffmpeg') is None:
        return

    with open(frames_path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        return
    first_ts = float(rows[0]['timestamp_sec'])
    last_ts = float(rows[-1]['timestamp_sec'])
    span = last_ts - first_ts
    if span <= 0:
        return
    true_fps = (len(rows) - 1) / span

    if abs(true_fps - config.CAM_VIDEO_FALLBACK_FPS) / true_fps < config.CAM_VIDEO_FPS_CORRECTION_THRESHOLD:
        return   # close enough to the provisional recording-time value — skip the remux

    fixed_path = video_path.with_name(video_path.stem + '_fixed' + video_path.suffix)
    result = subprocess.run(
        ['ffmpeg', '-y', '-r', f'{true_fps:.4f}', '-i', str(video_path), '-c', 'copy', str(fixed_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if result.returncode != 0 or not fixed_path.exists():
        print(f'[SESSION] video frame-rate correction failed, leaving camera_raw.avi as-recorded '
              f'(measured {true_fps:.2f}fps vs {config.CAM_VIDEO_FALLBACK_FPS:.2f}fps recorded): '
              f'{result.stdout[-500:]}')
        return
    fixed_path.replace(video_path)
    print(f'[SESSION] corrected camera_raw{config.CAM_VIDEO_EXT} frame rate '
          f'{config.CAM_VIDEO_FALLBACK_FPS:.2f} -> {true_fps:.2f} fps (measured from camera_frames.csv)')


def _extract_trial_videos(session_dir: Path, dataset_root: Path):
    """Crops the session's camera_raw.avi into a per-trial camera_raw.avi
    under each dataset/<label>/trial_XXX/, using camera_frames.csv's own
    per-frame timestamps to decide exactly which frames belong to each
    trial — deliberately NOT ffmpeg time-based seeking (-ss/-to), which was
    the previous approach here and is wrong for this data: -ss/-to seeks
    against the container's own internal timestamp track, which reflects
    whatever fps was declared when camera_raw.avi was written (see
    camera.py / _fix_video_framerate above). Even after that correction,
    the declared rate is a single session-wide average, not the true
    variable per-frame timing — real capture rate isn't constant
    (cap.read() + MediaPipe inference per frame varies), so a time-based
    seek can land on the wrong frames whenever the real local rate around a
    given trial differs from the session-wide average, which is exactly
    what produced trial clips with the wrong content. This is the same
    class of bug already fixed once for fingertip_imu.csv/trajectory.csv:
    trust each record's own explicit timestamp, never a single derived or
    assumed rate.

    frame_idx N in camera_raw.avi is exactly camera_frames.csv row N's
    frame — both are written in lockstep, one row per actually-written
    frame, by camera._video_writer_loop() (a frame dropped under
    backpressure never gets a CSV row either — see its docstring) — so
    matching by real timestamp here and then seeking to that literal frame
    index via cv2.CAP_PROP_POS_FRAMES (exact for an all-intra codec like
    MJPG — no keyframe-snapping ambiguity, unlike an inter-frame codec)
    gives the correct frames regardless of how uneven the real capture rate
    was anywhere in the session.

    Runs after cam_proc has fully exited (see run.py's _shutdown ordering),
    so camera_raw.avi/camera_frames.csv are guaranteed finalized and in
    sync. No ffmpeg dependency — unlike _fix_video_framerate above, this
    doesn't need it."""
    video_path = session_dir / f'camera_raw{config.CAM_VIDEO_EXT}'
    frames_path = session_dir / 'camera_frames.csv'
    if not video_path.exists() or not frames_path.exists():
        return
    with open(frames_path) as f:
        frame_ts = [float(r['timestamp_sec']) for r in csv.DictReader(f)]
    if not frame_ts:
        return

    metadata_path = dataset_root / 'metadata.csv'
    if not metadata_path.exists():
        return
    session_name = session_dir.name
    with open(metadata_path) as f:
        trials = [row for row in csv.DictReader(f) if row['session'] == session_name]
    if not trials:
        return

    import cv2 as _cv2   # only needed if there's actually something to extract

    cap = _cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f'[SESSION] could not open {video_path} for per-trial video extraction')
        return
    w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = _cv2.VideoWriter_fourcc(*config.CAM_VIDEO_CODEC)
    session_fps = cap.get(_cv2.CAP_PROP_FPS) or config.CAM_VIDEO_FALLBACK_FPS

    for trial in trials:
        label = trial['label']; trial_idx = int(trial['trial_idx'])
        trial_start = float(trial['start_sec']); trial_end = float(trial['end_sec'])
        trial_dir = dataset_root / label / f'trial_{trial_idx:03d}'
        if not trial_dir.exists():
            continue

        matching = [i for i, ts in enumerate(frame_ts) if trial_start <= ts <= trial_end]
        if not matching:
            print(f'[SESSION] no camera frames fell within {label}/trial_{trial_idx:03d}\'s '
                  f'[{trial_start:.3f}, {trial_end:.3f}]s window — skipping video for this trial')
            continue
        start_idx, end_idx = matching[0], matching[-1]

        # Trial-local average rate (from just this trial's own matched frames) is a better
        # playback-speed estimate for the cropped clip than the session-wide average — same
        # "trust the real local data over a global assumption" reasoning as the frame matching
        # above, just applied to the clip's own declared fps too.
        if end_idx > start_idx:
            trial_fps = (end_idx - start_idx) / (frame_ts[end_idx] - frame_ts[start_idx])
        else:
            trial_fps = session_fps

        out_path = trial_dir / f'camera_raw{config.CAM_VIDEO_EXT}'
        writer = _cv2.VideoWriter(str(out_path), fourcc, trial_fps, (w, h))
        cap.set(_cv2.CAP_PROP_POS_FRAMES, start_idx)
        for _ in range(start_idx, end_idx + 1):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()

    cap.release()


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