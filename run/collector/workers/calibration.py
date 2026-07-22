"""
collector/workers/calibration.py
────────────────────────────────────────────────────────────────────────────
Interactive index-fingertip trajectory calibration (see
trajectory_calibration.py), run once before each session starts, in its own
OS process — same rationale as unified_collector_final.py's
_run_precalibration_process: opening OpenCV's own GUI window in the main
process previously destabilized it once pynput's keyboard listener started
up right afterward (both touch low-level OS frameworks in the same
process). Running it here means the main process's own OpenCV usage stays
at zero, and the OS fully reclaims the camera device the moment this
subprocess exits, before camera.py's real capture process opens it.

Loads an existing run/calibration.json if present; press 'c' to
(re)calibrate, any other key to finish and let main() proceed to recording
with whatever calibration (if any) is active.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time


def _run_precalibration_process(camera_index: int, result_queue: "mp.Queue", mirror: bool = False):
    import cv2 as _cv2
    from fingertip_imu_multi import MultiFingertipIMUTracker
    from trajectory_calibration import load_calibration, run_calibration

    calibration = load_calibration()
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
            if mirror:
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
                new_calibration = run_calibration(tracker, frame, window_name)
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


def get_calibration(camera_index: int, skip: bool, mirror: bool = False) -> dict | None:
    """Called from run.py's main(), before the session starts and before
    camera.py's real capture process opens the device — so calibration time
    doesn't count toward the session clock and the two never fight over the
    camera. `skip` (--skip-calibration) silently reuses run/calibration.json
    if present instead of showing the interactive prompt. `mirror` must match
    whatever's passed to camera.py's camera_process_fn — the calibrated
    mic-anchored origin is only valid in the orientation it was measured in."""
    from trajectory_calibration import load_calibration

    if skip:
        return load_calibration()

    result_queue: "mp.Queue" = mp.Queue()
    proc = mp.Process(target=_run_precalibration_process, args=(camera_index, result_queue, mirror))
    proc.start()
    proc.join()
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        print('[CALIBRATE] calibration process exited without a result — continuing with any saved calibration')
        return load_calibration()
