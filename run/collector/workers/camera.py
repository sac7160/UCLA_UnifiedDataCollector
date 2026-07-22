"""
wristpad/workers/camera.py
────────────────────────────────────────────────────────────────────────────
Fingertip virtual IMU via a separate OS process (avoids GIL contention with
the audio callback, same rationale as the mic/watch-audio split). No live
preview frame is sent anywhere — this process's only job is tracking and
pushing IMU records to record_queue; camera_bridge_thread_fn() picks those
up and hands them to writers.write_fingertip_imu().

Each record is timestamped inside camera_process_fn(), right after
cap.read() and before running MediaPipe inference on it — i.e. at true
capture time, not whenever record_queue happens to get drained. Nothing
downstream (the queue, the bridge thread) ever needs to re-timestamp
anything.
"""

import multiprocessing as mp
import queue
import time

from ..core import state
from .writers import write_fingertip_imu


def camera_process_fn(camera_index: int, camera_pitch_deg, camera_roll_deg: float,
                       session_start: float,
                       record_queue: "mp.Queue", stop_flag: "mp.Event"):
    import cv2 as _cv2
    from fingertip_imu_multi import MultiFingertipIMUTracker, gravity_vector_from_camera_tilt
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
        frame = _cv2.flip(frame, 1)
        ts = time.perf_counter() - session_start
        records = tracker.update(frame, timestamp=ts)
        try:
            record_queue.put_nowait(records)
        except Exception:
            pass

    cap.release()
    tracker.close()


def camera_bridge_thread_fn(record_queue: "mp.Queue"):
    while not state.stop_event.is_set():
        try:
            records = record_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if records is None:
            continue
        try:
            write_fingertip_imu(records)
        except Exception:
            pass
