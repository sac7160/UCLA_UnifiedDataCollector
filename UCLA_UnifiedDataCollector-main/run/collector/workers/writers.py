"""
wristpad/workers/writers.py
────────────────────────────────────────────────────────────────────────────
The three "write this data to the session files (and push a copy to the
live display buffers)" functions, shared by watch_network.py (watch audio +
watch IMU) and camera.py (fingertip IMU).

All three accept a precise timestamp from the caller rather than sampling
"now" themselves — see each docstring for why that distinction matters. The
short version: these functions may run on a worker thread that's a little
behind, and if that lag ever leaks into the timestamp instead of just
delaying when the write happens, sync goes subtly wrong in a way that's
hard to notice until you overlay two supposedly-aligned files and the
peaks don't match.
"""

from __future__ import annotations

import numpy as np

from ..core import config, state
from ..core.utils import offset, rolling_cutoff
from trajectory_calibration import trajectory_csv_row


def write_watch_audio(raw_bytes: bytes, watch_ts_ms: float | None = None,
                       arrival_offset: float | None = None):
    """`arrival_offset`, if given, must be the offset()-domain timestamp of
    when this frame actually arrived over the socket — captured in the
    (fast) net recv thread, not here. This function always runs on a
    separate worker thread (see watch_network.watch_audio_worker_fn) so a
    slow disk write here can never delay draining the TCP socket; falling
    back to offset() here would silently re-introduce exactly that delay
    into the timestamp."""
    ts = arrival_offset if arrival_offset is not None else offset()
    samples = np.frombuffer(raw_bytes, dtype='<i2').astype(np.float32)
    state.watch_rms = float(np.sqrt(np.mean(samples ** 2)))
    with state.file_lock:
        if state.watch_audio_offset is None:
            state.watch_audio_offset = ts
            state.sync['watch_audio_offset_sec'] = state.watch_audio_offset
    if state.watch_wf:
        state.watch_wf.writeframes(raw_bytes)

    n_samples = len(raw_bytes) // 2
    if state.watch_audio_frames_writer:
        state.watch_audio_frames_writer.writerow([
            state.watch_audio_session_samples, n_samples,
            f'{watch_ts_ms:.0f}' if watch_ts_ms else '',
        ])
    state.watch_audio_session_samples += n_samples

    with state.rolling_lock:
        state.watch_audio_rolling.append((ts, raw_bytes, watch_ts_ms))
        cutoff = rolling_cutoff(ts)
        while state.watch_audio_rolling and state.watch_audio_rolling[0][0] < cutoff:
            state.watch_audio_rolling.popleft()

    if state.disp_watch_wave is not None:
        norm = samples / 32768.0
        state.disp_watch_wave.push(norm)
        state.disp_watch_spec.push(norm)


def write_imu(sensor: str, v1: float, v2: float, v3: float, watch_ts_ms: float = 0.0,
              display_ts: float | None = None, arrival_offset: float | None = None):
    """`arrival_offset`, like in write_watch_audio above, should be the true
    socket-arrival time captured in the net recv thread — this function
    runs on the (separately delayable) watch-IMU worker thread."""
    ts = arrival_offset if arrival_offset is not None else offset()
    with state.file_lock:
        if state.imu_offset is None:
            state.imu_offset = ts
            state.sync['imu_offset_sec'] = state.imu_offset
        if state.imu_writer:
            state.imu_writer.writerow([f'{ts:.6f}', sensor, f'{v1:.6f}', f'{v2:.6f}', f'{v3:.6f}',
                                        f'{watch_ts_ms:.0f}'])
            state.imu_flush_n += 1
            if state.imu_flush_n % 40 == 0 and state.imu_fp:
                state.imu_fp.flush()

    with state.rolling_lock:
        state.imu_rolling.append((ts, sensor, v1, v2, v3, watch_ts_ms))
        cutoff = rolling_cutoff(ts)
        while state.imu_rolling and state.imu_rolling[0][0] < cutoff:
            state.imu_rolling.popleft()

    target = state.disp_watch_acc if sensor == 'acc' else state.disp_watch_gyro
    target.push(v1, v2, v3, ts=display_ts)   # display_ts=None falls back to time.perf_counter()


def write_fingertip_imu(records: list):
    """records: list[FingertipIMURecord] (5 fingers). Each record's own
    .timestamp (stamped in the camera process at true capture time — see
    camera.py) is used throughout; nothing here samples a fresh "now"."""
    with state.file_lock:
        if state.cam_offset is None:
            state.cam_offset = offset()
            state.sync['fingertip_imu_offset_sec'] = state.cam_offset
        if state.cam_writer:
            for r in records:
                state.cam_writer.writerow([
                    f'{r.timestamp:.6f}', r.finger, r.hand_label, int(r.detected),
                    f'{r.accel_x:.4f}', f'{r.accel_y:.4f}', f'{r.accel_z:.4f}',
                    f'{r.gyro_x:.6f}', f'{r.gyro_y:.6f}', f'{r.gyro_z:.6f}',
                    f'{r.pos_x:.3f}', f'{r.pos_y:.3f}', f'{r.pos_z:.3f}',
                ])
            state.cam_flush_n += 1
            if state.cam_flush_n % config.CAM_FLUSH_EVERY_N == 0 and state.cam_fp:
                state.cam_fp.flush()

    with state.trial_lock:
        if state.trial_active:
            state.trial_buffers['fingertip'].extend(records)

    for r in records:
        if r.finger != state.display_finger or not r.detected:
            continue
        acc_vals = (r.accel_x, r.accel_y, r.accel_z)
        gyro_vals = (r.gyro_x, r.gyro_y, r.gyro_z)
        if all(np.isfinite(v) for v in acc_vals):
            state.disp_finger_acc.push(*acc_vals)
        if all(np.isfinite(v) for v in gyro_vals):
            state.disp_finger_gyro.push(*gyro_vals)


def write_trajectory(traj: dict):
    """traj: dict returned by trajectory_calibration.compute_trajectory() for
    one frame (same frame whose records were just passed to
    write_fingertip_imu() — see camera.py). Uses the index record's own
    timestamp, same as write_fingertip_imu does for its rows."""
    ts = traj['index_record'].timestamp
    with state.file_lock:
        if state.traj_writer:
            state.traj_writer.writerow(trajectory_csv_row(ts, traj))
            state.traj_flush_n += 1
            if state.traj_flush_n % config.CAM_FLUSH_EVERY_N == 0 and state.traj_fp:
                state.traj_fp.flush()

    with state.trial_lock:
        if state.trial_active:
            state.trial_buffers['trajectory'].append(traj)

    if state.disp_trajectory is not None:
        state.disp_trajectory.push(traj)
