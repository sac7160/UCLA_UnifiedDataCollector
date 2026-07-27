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

Raw video recording (record_video=True, the default): every captured frame
(post-mirror, pre-MediaPipe) is also handed to a dedicated writer thread
that owns an MJPG cv2.VideoWriter + a camera_frames.csv sidecar for the
whole session — see _video_writer_loop()'s docstring for why this lives on
its own thread and why it's MJPG specifically. This writes one continuous
camera_raw.avi per session, same as every other stream in this pipeline
(imu.csv, trajectory.csv, both WAVs) — REC on/off never gates what's
captured here, only what session.py's _extract_trial_videos() later crops
out of it into each trial's folder, after the session closes.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time

from ..core import state
from .writers import write_fingertip_imu, write_trajectory


def _video_writer_loop(video_queue: "queue.Queue", session_dir, codec: str, ext: str,
                        fps: float, flush_every_n: int):
    """Drains video_queue and is the *only* thing that ever touches
    cv2.VideoWriter or camera_frames.csv — runs on its own thread inside the
    camera subprocess so a slow disk/encode call can never delay
    cap.read()/MediaPipe in camera_process_fn's loop, the same "capture path
    never blocked by disk I/O" pattern already used for
    touch_detection.mic_wav_writer_fn / watch_network.watch_audio_worker_fn.

    frame_idx in camera_frames.csv is assigned here, in write order — a
    frame dropped upstream under backpressure (video_queue full) never
    reaches this thread, so it never gets a CSV row either; the CSV always
    describes exactly what's in camera_raw.avi, which is what a later
    ffmpeg-based crop (session.py's _extract_trial_videos) or manual review
    needs. The VideoWriter is opened lazily on the first real frame, using
    its actual shape — cap.get(CAP_PROP_FRAME_WIDTH/HEIGHT) can return 0 or
    a stale value before the device has actually started streaming.

    A None item on video_queue is the shutdown sentinel: release() the
    writer (required for the file to be valid — an unfinalized video file
    can be unreadable) and close the CSV, then return.
    """
    import cv2 as _cv2
    import csv as _csv

    writer = None
    frames_fp = open(session_dir / 'camera_frames.csv', 'w', newline='')
    frames_writer = _csv.writer(frames_fp)
    frames_writer.writerow(['frame_idx', 'timestamp_sec'])
    frame_idx = 0
    fourcc = _cv2.VideoWriter_fourcc(*codec)

    try:
        while True:
            item = video_queue.get()
            if item is None:
                break
            frame, ts = item
            if writer is None:
                h, w = frame.shape[:2]
                writer = _cv2.VideoWriter(str(session_dir / f'camera_raw{ext}'), fourcc, fps, (w, h))
            writer.write(frame)
            frames_writer.writerow([frame_idx, f'{ts:.6f}'])
            frame_idx += 1
            if frame_idx % flush_every_n == 0:
                frames_fp.flush()
    finally:
        if writer is not None:
            writer.release()
        frames_fp.flush()
        frames_fp.close()


def camera_process_fn(camera_index: int, camera_pitch_deg, camera_roll_deg: float,
                       session_start_wall: float, session_dir,
                       record_queue: "mp.Queue", stop_flag: "mp.Event",
                       calibration: dict | None = None, mirror: bool = False,
                       record_video: bool = True):
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

    video_queue = None
    writer_thread = None
    if record_video:
        # NOT cap.get(CAP_PROP_FPS): queried this early (before any frame has
        # actually been read) it's commonly 0 or a stale/nominal value that
        # doesn't reflect the real achieved rate once MediaPipe inference runs
        # in this same loop every frame — and getting it right here doesn't
        # matter anyway, since session.py's _fix_video_framerate() relabels
        # the container with the true measured fps (from camera_frames.csv's
        # own per-frame timestamps) once the session ends. This is only ever
        # a provisional value for the live VideoWriter to be opened with.
        fps = config.CAM_VIDEO_FALLBACK_FPS
        video_queue = queue.Queue(maxsize=config.CAM_VIDEO_QUEUE_MAXSIZE)
        writer_thread = threading.Thread(
            target=_video_writer_loop,
            args=(video_queue, session_dir, config.CAM_VIDEO_CODEC, config.CAM_VIDEO_EXT,
                  fps, config.CAM_FLUSH_EVERY_N),
            daemon=True,
        )
        writer_thread.start()

    video_dropped = 0
    while not stop_flag.is_set():
        success, frame = cap.read()
        if not success:
            continue
        if mirror:
            frame = _cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts = time.time() - session_start_wall

        if video_queue is not None:
            # .copy(): frame is about to be handed to a different thread
            # (tracker.update() below only reads it via cv2.cvtColor, which
            # already allocates a new array — this copy isn't strictly
            # required by today's code, but it's cheap (~1ms at 720p) insurance
            # against that invariant silently breaking later.
            try:
                video_queue.put_nowait((frame.copy(), ts))
            except queue.Full:
                video_dropped += 1

        records = tracker.update(frame, timestamp=ts)
        traj = compute_trajectory(tracker, records, w, h, calibration)
        try:
            record_queue.put_nowait((records, traj, video_dropped))
        except Exception:
            pass

    if video_queue is not None:
        video_queue.put(None)
        writer_thread.join(timeout=config.CAM_VIDEO_DRAIN_TIMEOUT_SEC)
        if writer_thread.is_alive():
            print('[CAMERA] video writer thread did not finish flushing in time — '
                  'camera_raw file may be missing trailing frames')

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
        records, traj, video_dropped = payload
        state.video_frames_dropped = video_dropped
        try:
            write_fingertip_imu(records)
            write_trajectory(traj)
        except Exception:
            pass
