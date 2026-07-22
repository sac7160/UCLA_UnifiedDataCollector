"""
wristpad/workers/camera.py
────────────────────────────────────────────────────────────────────────────
Fingertip virtual IMU + index-finger trajectory, via a separate OS process
(avoids GIL contention with the audio callback, same rationale as the
mic/watch-audio split). No live preview frame is sent anywhere — this
process's only job is tracking and pushing (records, traj) pairs to
record_queue; camera_bridge_thread_fn() picks those up and hands them to
writers.write_fingertip_imu() / writers.write_trajectory().

Each record/traj pair is timestamped inside camera_process_fn(), right after
cap.read() and before running MediaPipe inference on it — i.e. at true
capture time, not whenever record_queue happens to get drained. Nothing
downstream (the queue, the bridge thread) ever needs to re-timestamp
anything.

Timestamps are anchored via time.time() (wall clock, seconds since epoch),
NOT time.perf_counter() — perf_counter's docs only guarantee monotonicity
*within* a process, not a shared epoch across processes, and in practice on
this codebase's target platforms a perf_counter() value computed in this
subprocess and differenced against a perf_counter() value captured in the
main process (session.py's state.session_start) can be off by tens of
seconds. time.time() means the same real-world instant in every process on
the machine, so subtracting the wall-clock reference captured in the main
process (state.session_start_wall) keeps this timestamp on the same
timeline as the main process's offset() — which is what trial.py's
process_trial() compares fingertip/trajectory timestamps against when
cropping a trial window.

Trajectory computation (compute_trajectory, from trajectory_calibration.py)
reuses the same MediaPipe tracker state as the fingertip IMU records, so it
costs no extra inference — just a bit of extra arithmetic per frame. The
`calibration` dict is loaded once in run.py's main() (see
collector/workers/calibration.py) and passed in here unchanged for the
lifetime of the process; recalibrating requires restarting collection.

`mirror` (--mirror) is off by default: MediaPipe's handedness classification
and every x-coordinate (x_px, local_x_mm, global_x_mm, ...) are computed
directly off whatever frame is passed to the tracker, so mirroring it here
would mirror all of that too, relative to true physical left/right. Must be
passed the same way to calibration.py's precalibration step (see run.py's
main()) — the calibrated mic-anchored origin is only valid in whichever
orientation it was measured in.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time

from ..core import state
from .writers import write_fingertip_imu, write_trajectory


def camera_process_fn(camera_index: int, camera_pitch_deg, camera_roll_deg: float,
                       session_start_wall: float,
                       record_queue: "mp.Queue", stop_flag: "mp.Event",
                       calibration: dict | None = None, mirror: bool = False):
    import cv2 as _cv2
    from fingertip_imu_multi import MultiFingertipIMUTracker, gravity_vector_from_camera_tilt
    from trajectory_calibration import compute_trajectory
    from ..core import config

    gravity_mm_s2 = None
    if camera_pitch_deg is not None:
        gravity_mm_s2 = gravity_vector_from_camera_tilt(camera_pitch_deg, camera_roll_deg)

    tracker = MultiFingertipIMUTracker(
        max_num_hands=1, smoothing_window=config.CAM_SMOOTHING_WINDOW,
        gravity_mm_s2=gravity_mm_s2, ema_alpha=config.CAM_EMA_ALPHA,
    )
    cap = _cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        record_queue.put(None)
        return

    while not stop_flag.is_set():
        success, frame = cap.read()
        if not success:
            continue
        if mirror:
            frame = _cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts = time.time() - session_start_wall
        records = tracker.update(frame, timestamp=ts)
        traj = compute_trajectory(tracker, records, w, h, calibration)
        try:
            record_queue.put_nowait((records, traj))
        except Exception:
            pass

    cap.release()
    tracker.close()


def camera_bridge_thread_fn(record_queue: "mp.Queue"):
    while not state.stop_event.is_set():
        try:
            payload = record_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if payload is None:
            continue
        records, traj = payload
        try:
            write_fingertip_imu(records)
            write_trajectory(traj)
        except Exception:
            pass
