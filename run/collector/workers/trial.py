"""
data_collector/workers/trial.py
────────────────────────────────────────────────────────────────────────────
Trial boundaries:

  - toggle_recording() is the REC gate — press once to start, again to
    stop (spacebar or the instructor window's button both call this).
  - The saved trial always spans the full [1st spacebar press, 2nd
    spacebar press] window — it is never trimmed to touch-on/touch-off.
  - Within that window, touch_detection.py's audio worker still calls
    write_event_at('audio_touch_on'/'audio_touch_off', ...) every time the
    detector fires, so every individual touch event's precise,
    debounce-backdated timestamp is still in events.csv alongside
    rec_start/rec_end — just as data about the trial, not as something
    that changes what gets saved.

process_trial() does the actual cropping: given a (start, end) window and
a snapshot of the buffered streams, it slices each stream down to that
window (RTBGN-mapped where possible) and writes trial_XXX/ under
dataset/<label>/.
"""

import csv
import json
import queue
import time
from datetime import datetime
from math import gcd

import numpy as np
import scipy.io.wavfile as wavfile
from pynput import keyboard
from scipy.signal import resample_poly

from ..core import config, state
from ..core.utils import log, offset
from trajectory_calibration import TRAJECTORY_CSV_HEADER, trajectory_csv_row


def write_event(event: str):
    ts = offset()
    with state.file_lock:
        if state.events_writer:
            state.events_writer.writerow([f'{ts:.6f}', event])
            state.events_fp.flush()
        state.event_log.append((ts, event))
    log('EVENT', f'{event}  offset={ts:.4f}s')


def write_event_at(event: str, ts: float):
    """Same as write_event, but for a caller-supplied (already backdated)
    timestamp rather than "now" — used for audio-detected events, since the
    debounce backdating means the true event time is earlier than the
    moment it got confirmed."""
    with state.file_lock:
        if state.events_writer:
            state.events_writer.writerow([f'{ts:.6f}', event])
            state.events_fp.flush()
        state.event_log.append((ts, event))
    log('EVENT', f'{event}  offset={ts:.4f}s (backdated)')


def toggle_recording():
    """REC start/stop = the only thing that decides what gets saved. The
    saved trial always spans [1st spacebar press, 2nd spacebar press] —
    touch_detection.py's audio worker still logs every individual
    audio_touch_on/audio_touch_off to events.csv regardless (see
    touch_detection.audio_worker_fn), but those no longer trim the saved
    trial's boundaries; they're just timestamps recorded alongside it."""
    if not state.rec_active:
        # state.current_label/current_stimulus are kept up to date live by
        # the instructor window's class-picker dropdown — nothing to pull
        # here at REC-start anymore, they're already whatever was last
        # selected.
        write_event('rec_start')
        with state.trial_lock:
            for k in state.trial_buffers:
                state.trial_buffers[k].clear()
            state.trial_start_offset = offset()
            state.trial_active = True
        state.rec_active = True
        state.touch_candidate_on_time = 0.0
        state.touch_candidate_off_time = 0.0
        if state.touch_on_state:
            # Edge case: detector was already reading ON right as REC
            # started — no earlier "true onset" available, so the press
            # itself is the onset.
            state.audio_touch_start = offset()
            write_event_at('audio_touch_on', state.audio_touch_start)
    else:
        release_offset = offset()
        if state.audio_touch_start is not None:
            # Edge case: a touch was still ongoing when REC stopped — log
            # its end at the stop moment rather than leaving it unclosed.
            write_event_at('audio_touch_off', release_offset)
            state.audio_touch_start = None
        state.rec_active = False

        with state.trial_lock:
            state.trial_active = False
            rec_start = state.trial_start_offset
            snapshot = {k: list(v) for k, v in state.trial_buffers.items()}
        write_event('rec_end')

        label = state.current_label
        if rec_start is not None and release_offset > rec_start:
            with state.pending_lock:
                state.pending_starts.append(rec_start)
            state.trial_queue.put((rec_start, release_offset, snapshot, 'spacebar', label))


def on_key_press(key):
    if key == keyboard.Key.space and not state.space_down:
        state.space_down = True
        toggle_recording()
    elif key == keyboard.Key.esc:
        state.stop_event.set()
        return False


def on_key_release(key):
    if key == keyboard.Key.space:
        state.space_down = False


def trial_worker_fn():
    while not state.stop_event.is_set() or not state.trial_queue.empty():
        try:
            start, end, snapshot, trigger, label = state.trial_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        try:
            time.sleep(max(config.IMU_GRACE_SEC, config.WATCH_AUDIO_GRACE_SEC))

            with state.rolling_lock:
                snapshot['imu']         = list(state.imu_rolling)
                snapshot['watch_audio'] = list(state.watch_audio_rolling)

            with state.pending_lock:
                if start in state.pending_starts:
                    state.pending_starts.remove(start)

            process_trial(start, end, snapshot, trigger, label)
        except Exception as e:
            log('TRIAL', f'error while processing: {e}')


def crop_watch_audio_frames(frames, trial_start: float, trial_end: float, sr: int):
    """Sample-accurate crop of watch-audio frames to [trial_start, trial_end].

    `frames` is an iterable of (frame_start_pc, int16 sample array) — the
    frame's already latency-corrected estimated start time in PC seconds,
    and its raw samples. This is the ONE place that decides which samples
    make it into a trial's watch_audio.wav; both the live per-trial save
    (process_trial, below) and the post-session recalibration
    (session._recalibrate_session_trials) call this, so there's no risk of
    the two drifting apart and silently reintroducing the frame-level (not
    sample-level) cropping slop this replaced — see the chat this was
    unified in for how that divergence showed up as confusing timing
    results after changing WATCH_AUDIO_LATENCY_SEC.

    Each watch-audio frame spans ~40ms; including/excluding a whole frame
    based only on its start time could leave up to ~40ms of slop at each
    trial boundary vs. the much finer-grained surface mic (2.7ms blocks) —
    slicing each frame down to just the samples that actually fall inside
    the window closes that gap to single-sample precision.
    """
    pieces = []
    for frame_start_pc, frame_samples in frames:
        n = len(frame_samples)
        frame_end_pc = frame_start_pc + n / sr
        if frame_end_pc < trial_start or frame_start_pc > trial_end:
            continue   # entirely outside the trial window
        sample_times = frame_start_pc + np.arange(n) / sr
        mask = (sample_times >= trial_start) & (sample_times <= trial_end)
        if np.any(mask):
            pieces.append((frame_start_pc, frame_samples[mask]))
    pieces.sort(key=lambda item: item[0])   # frames can arrive out of capture order under network jitter
    if not pieces:
        return np.array([], dtype=np.int16)
    return np.concatenate([s for _, s in pieces])


def process_trial(start: float, end: float, snapshot: dict, trigger: str, label: str):
    """Crops the buffered snapshot and saves it in dataset/<label>/
    trial_XXX/. `start`/`end` are the REC start/stop offsets — the saved
    trial always spans the full spacebar-to-spacebar window; `trigger` is
    currently always 'spacebar', kept as a column for any future variant."""
    margin = state.trial_margin
    trial_start = start + margin
    trial_end   = end - margin
    if trial_end <= trial_start:
        return

    rtbgn_watch_ms = state.sync.get('rtbgn_watch_ms')
    rtbgn_pc_sec   = state.sync.get('rtbgn_pc_sec')
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

    traj_rel = [
        (traj['index_record'].timestamp - trial_start, traj)
        for traj in snapshot.get('trajectory', [])
        if trial_start <= traj['index_record'].timestamp <= trial_end
    ]

    if not imu_rel and not ft_rel:
        return

    raw_frames = []
    for (ts, chunk, watch_ts_ms) in snapshot['watch_audio']:
        if use_rtbgn and watch_ts_ms:
            frame_start_pc = (watch_ts_ms - rtbgn_watch_ms) / 1000.0 + rtbgn_pc_sec - config.WATCH_AUDIO_LATENCY_SEC
        else:
            frame_start_pc = ts - config.WATCH_AUDIO_LATENCY_SEC
        raw_frames.append((frame_start_pc, np.frombuffer(chunk, dtype='<i2')))
    wa_samples = crop_watch_audio_frames(raw_frames, trial_start, trial_end, config.WATCH_AUDIO_SR)

    mic_blocks = [blk for (ts, blk) in snapshot['mic'] if trial_start <= ts <= trial_end]
    if mic_blocks:
        mic_concat = np.concatenate(mic_blocks).astype(np.float64)
        if state.mic_sr_runtime != config.WATCH_AUDIO_SR:
            g = gcd(config.WATCH_AUDIO_SR, state.mic_sr_runtime)
            mic_rs = resample_poly(mic_concat, config.WATCH_AUDIO_SR // g, state.mic_sr_runtime // g)
        else:
            mic_rs = mic_concat
    else:
        mic_rs = np.array([], dtype=np.float64)
    mic_int16 = np.clip(mic_rs * 32767, -32768, 32767).astype(np.int16)

    label_use = label if label else 'unlabeled'
    label_dir = state.trial_dataset_root / label_use
    label_dir.mkdir(parents=True, exist_ok=True)
    trial_idx = len(sorted(label_dir.glob('trial_*'))) + 1
    trial_dir = label_dir / f'trial_{trial_idx:03d}'
    trial_dir.mkdir(parents=True, exist_ok=True)

    wavfile.write(trial_dir / 'watch_audio.wav', config.WATCH_AUDIO_SR, wa_samples)
    wavfile.write(trial_dir / 'surface_mic.wav', config.WATCH_AUDIO_SR, mic_int16)

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

    # Same time_aligned convention as imu.csv/fingertip_imu.csv above (0 =
    # this trial's start) — session-level events.csv uses offset-from-
    # session-start instead, which isn't directly comparable to the other
    # files sitting in this same trial folder; this re-expresses whichever
    # of them fall inside [trial_start, trial_end] on the trial's own clock.
    with state.file_lock:
        events_in_trial = [(ts, ev) for ts, ev in state.event_log if trial_start <= ts <= trial_end]
    with open(trial_dir / 'events.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_aligned', 'event'])
        for ts, ev in sorted(events_in_trial, key=lambda row: row[0]):
            w.writerow([f'{ts - trial_start:.6f}', ev])

    metadata_path = state.trial_dataset_root / 'metadata.csv'
    write_header = not metadata_path.exists()
    with open(metadata_path, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(['session', 'label', 'trial_idx', 'start_sec', 'end_sec',
                        'duration_sec', 'margin_sec', 'trigger', 'material'])
        w.writerow([
            state.session_dir.name if state.session_dir else '',
            label_use, trial_idx,
            f'{trial_start:.6f}', f'{trial_end:.6f}',
            f'{trial_end - trial_start:.6f}', margin, trigger, state.current_material,
        ])

    # Same info as the metadata.csv row above, but living *inside* the
    # trial's own folder — so opening trial_XXX/ alone (without cross-
    # referencing the dataset-wide metadata.csv) is enough to know when it
    # was collected and what material was active. collected_at is a real
    # wall-clock timestamp: session_start_epoch (time.time(), captured at
    # start_session()) plus trial_start (a perf_counter()-based offset from
    # that same moment) — see session.py's sync dict.
    session_start_epoch = state.sync.get('session_start_epoch')
    if session_start_epoch is not None:
        collected_at = datetime.fromtimestamp(session_start_epoch + trial_start).strftime('%Y-%m-%d %H:%M:%S')
    else:
        collected_at = None
    trial_info = {
        'session': state.session_dir.name if state.session_dir else '',
        'label': label_use,
        'trial_idx': trial_idx,
        'trigger': trigger,
        'material': state.current_material,
        'collected_at': collected_at,
        'start_sec': round(trial_start, 6),
        'end_sec': round(trial_end, 6),
        'duration_sec': round(trial_end - trial_start, 6),
        'margin_sec': margin,
    }
    with open(trial_dir / 'trial_info.json', 'w') as f:
        json.dump(trial_info, f, indent=2)