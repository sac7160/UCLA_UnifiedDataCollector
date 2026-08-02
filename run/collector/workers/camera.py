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
cap.read() and before handing the frame off for MediaPipe inference — i.e.
at true capture time, not whenever record_queue happens to get drained, and
not whenever the tracking worker thread (see below) actually gets around to
running inference on it. Nothing downstream (either queue, either worker
thread, the bridge thread) ever needs to re-timestamp anything.

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

CAPTURE / TRACKING SPLIT — why the loop that calls cap.read() never calls
tracker.update() itself: MediaPipe inference time varies frame to frame
(hand pose, lighting, occlusion, how many candidate detections it has to
evaluate), and running it inline in the same loop as cap.read() means a
single slow-inference frame directly delays the next cap.read() call. On
this codebase's target camera/OS combination, a delayed cap.read() doesn't
just arrive late — the driver's own small ring buffer silently drops
whatever frame(s) arrived while the loop was busy, and how often that
happens tracks how often inference happens to run slow, not any single
fixed rate. Measured fingertip frame-drop rates on trials collected before
this split ranged from 0% to over 20% run to run, which matches "however
long inference happened to take that trial" far better than a hardware
frame-rate ceiling. The fix mirrors _video_writer_loop's existing rationale
("capture path never blocked by disk I/O") one level further: a dedicated
_tracking_worker_loop thread is now the only thing that ever calls
tracker.update()/compute_trajectory(), fed by a small bounded queue, so the
main loop's only job is calling cap.read() as fast as the camera will
deliver frames, regardless of how long inference on any given frame takes.

Trajectory computation (compute_trajectory, from trajectory_calibration.py)
reuses the same MediaPipe tracker state as the fingertip IMU records, so it
costs no extra inference — just a bit of extra arithmetic per frame. Both
now run on the tracking worker thread, not the capture loop, for the same
reason as tracker.update() above. The `calibration` dict is loaded once in
run.py's main() (see collector/workers/calibration.py) and passed in here
unchanged for the lifetime of the process; recalibrating requires
restarting collection.

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

DROP ACCOUNTING: there are now three independent points a frame's data can
be dropped at under backpressure, each counted separately since they
indicate different bottlenecks:
  video_dropped     capture loop -> video writer thread (video_queue full,
                     i.e. disk/encode falling behind)
  tracking_dropped   capture loop -> tracking worker thread (tracking_queue
                     full, i.e. MediaPipe inference falling behind — this
                     is the one the capture/tracking split above targets)
  record_dropped     tracking worker thread -> bridge thread, i.e. across
                     the process boundary (record_queue full, i.e. the main
                     process — GUI, audio, watch-network threads — falling
                     behind draining it)
All three are threaded through to camera_bridge_thread_fn and exposed on
`state`, mirroring the state.video_frames_dropped pattern that already
existed for the video path alone.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import threading
import time

from ..core import state
from .writers import write_fingertip_imu, write_trajectory


def _video_writer_loop(video_queue: "queue.Queue", session_dir, codec: str, ext: str,
                        fps: float, flush_every_n: int, video_stem: str, frames_csv_name: str):
    """Drains video_queue and is the *only* thing that ever touches
    cv2.VideoWriter or this camera's own frames CSV — runs on its own
    thread so a slow disk/encode call can never delay cap.read()/MediaPipe
    in the capture loop that feeds it, the same "capture path never
    blocked by disk I/O" pattern already used for
    touch_detection.mic_wav_writer_fn / watch_network.watch_audio_worker_fn.
    Shared verbatim by both cameras (camera_process_fn for the primary
    camera, camera2_thread_fn for the video-only second one) — video_stem/
    frames_csv_name are the only thing that ever differs between them
    (config.CAM_VIDEO_STEM/CAM_FRAMES_CSV vs. CAM2_VIDEO_STEM/CAM2_FRAMES_CSV),
    so the two cameras' output can never collide on the same filenames.

    frame_idx in <frames_csv_name> is assigned here, in write order — a
    frame dropped upstream under backpressure (video_queue full) never
    reaches this thread, so it never gets a CSV row either; the CSV always
    describes exactly what's in <video_stem><ext>, which is what a later
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
    frames_fp = open(session_dir / frames_csv_name, 'w', newline='')
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
                writer = _cv2.VideoWriter(str(session_dir / f'{video_stem}{ext}'), fourcc, fps, (w, h))
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


def _tracking_worker_loop(tracking_queue: "queue.Queue", tracker, calibration: dict | None,
                           record_queue: "mp.Queue"):
    """Drains tracking_queue and is the *only* thing that ever calls
    tracker.update()/compute_trajectory() — see this module's docstring
    ("CAPTURE / TRACKING SPLIT") for why inference was moved off the
    capture loop entirely. record_dropped counts frames whose tracking
    result couldn't be handed across the process boundary (record_queue
    full — the main process falling behind), separately from
    tracking_dropped (counted in camera_process_fn: frames that never even
    made it INTO this thread's queue, i.e. this thread itself falling
    behind) — the two indicate different bottlenecks and shouldn't be
    conflated into one number.

    A None item on tracking_queue is the shutdown sentinel, mirroring
    _video_writer_loop's convention.
    """
    from trajectory_calibration import compute_trajectory

    record_dropped = 0
    while True:
        item = tracking_queue.get()
        if item is None:
            break
        frame, ts, video_dropped, tracking_dropped = item
        h, w = frame.shape[:2]
        records = tracker.update(frame, timestamp=ts)
        traj = compute_trajectory(tracker, records, w, h, calibration)
        try:
            record_queue.put_nowait((records, traj, video_dropped, tracking_dropped, record_dropped))
        except queue.Full:
            record_dropped += 1


def camera_process_fn(camera_index: int, camera_pitch_deg, camera_roll_deg: float,
                       session_start_wall: float, session_dir,
                       record_queue: "mp.Queue", stop_flag: "mp.Event",
                       calibration: dict | None = None, mirror: bool = False,
                       record_video: bool = True):
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

    # Requested BEFORE any cap.read() — a UVC camera picks its actual mode
    # at the first read otherwise, and changing width/height/fps after
    # frames have already started flowing can silently fail or force a
    # brief re-init. Not guaranteed to be honored exactly (the driver
    # snaps to its nearest supported mode), so the actually-negotiated
    # values are read back and logged right after — never assume the
    # request took effect without checking.
    cap.set(_cv2.CAP_PROP_FRAME_WIDTH, config.CAM_CAPTURE_WIDTH)
    cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_CAPTURE_HEIGHT)
    cap.set(_cv2.CAP_PROP_FPS, config.CAM_CAPTURE_FPS_REQUEST)
    actual_w = cap.get(_cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(_cv2.CAP_PROP_FPS)
    print(f'[CAMERA] requested {config.CAM_CAPTURE_WIDTH}x{config.CAM_CAPTURE_HEIGHT}'
          f'@{config.CAM_CAPTURE_FPS_REQUEST}fps — camera reports {actual_w:.0f}x{actual_h:.0f}'
          f'@{actual_fps:.1f}fps (driver-reported FPS is a nominal value, not a measured one — '
          f'raw_camera_rate_test.py / camera_frames.csv are what confirm the real achieved rate)')

    video_queue = None
    writer_thread = None
    if record_video:
        # NOT cap.get(CAP_PROP_FPS): queried this early (before any frame has
        # actually been read) it's commonly 0 or a stale/nominal value that
        # doesn't reflect the real achieved rate — and getting it right here
        # doesn't matter anyway, since session.py's _fix_video_framerate()
        # relabels the container with the true measured fps (from
        # camera_frames.csv's own per-frame timestamps) once the session
        # ends. This is only ever a provisional value for the live
        # VideoWriter to be opened with.
        fps = config.CAM_VIDEO_FALLBACK_FPS
        video_queue = queue.Queue(maxsize=config.CAM_VIDEO_QUEUE_MAXSIZE)
        writer_thread = threading.Thread(
            target=_video_writer_loop,
            args=(video_queue, session_dir, config.CAM_VIDEO_CODEC, config.CAM_VIDEO_EXT,
                  fps, config.CAM_FLUSH_EVERY_N, config.CAM_VIDEO_STEM, config.CAM_FRAMES_CSV),
            daemon=True,
        )
        writer_thread.start()

    # See this module's docstring ("CAPTURE / TRACKING SPLIT") — tracker.update()/
    # compute_trajectory() no longer run in this loop. tracking_queue's maxsize
    # is deliberately small (config.CAM_TRACKING_QUEUE_MAXSIZE, expected on the
    # order of a handful of frames): a MediaPipe inference that's genuinely
    # falling behind should show up as counted drops promptly, not as several
    # seconds of growing latency silently queued up before anything is dropped.
    tracking_queue: "queue.Queue" = queue.Queue(maxsize=config.CAM_TRACKING_QUEUE_MAXSIZE)
    tracking_thread = threading.Thread(
        target=_tracking_worker_loop,
        args=(tracking_queue, tracker, calibration, record_queue),
        daemon=True,
    )
    tracking_thread.start()

    video_dropped = 0
    tracking_dropped = 0
    while not stop_flag.is_set():
        success, frame = cap.read()
        if not success:
            continue
        if mirror:
            frame = _cv2.flip(frame, 1)
        ts = time.time() - session_start_wall

        if video_queue is not None:
            # .copy(): frame is about to be handed to another thread (the
            # video writer). See tracking_queue's .copy() just below for the
            # same reasoning applied to the tracking path — each queue's
            # consumer runs independently and at its own pace, so each needs
            # its own frame, not a shared reference to the same array.
            try:
                video_queue.put_nowait((frame.copy(), ts))
            except queue.Full:
                video_dropped += 1

        try:
            tracking_queue.put_nowait((frame.copy(), ts, video_dropped, tracking_dropped))
        except queue.Full:
            tracking_dropped += 1

    if video_queue is not None:
        video_queue.put(None)
        writer_thread.join(timeout=config.CAM_VIDEO_DRAIN_TIMEOUT_SEC)
        if writer_thread.is_alive():
            print('[CAMERA] video writer thread did not finish flushing in time — '
                  'camera_raw file may be missing trailing frames')

    tracking_queue.put(None)
    tracking_thread.join(timeout=config.CAM_TRACKING_DRAIN_TIMEOUT_SEC)
    if tracking_thread.is_alive():
        print('[CAMERA] tracking worker thread did not finish in time — '
              'trailing fingertip IMU/trajectory records may be missing')

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
        records, traj, video_dropped, tracking_dropped, record_dropped = payload
        state.video_frames_dropped = video_dropped
        state.tracking_frames_dropped = tracking_dropped
        state.record_frames_dropped = record_dropped
        try:
            write_fingertip_imu(records)
            write_trajectory(traj)
        except Exception:
            pass


def camera2_thread_fn(camera_index: int, session_start_wall: float, session_dir,
                       stop_event: "threading.Event"):
    """Second, independent camera — video-only, no MediaPipe tracking.
    Unlike camera_process_fn (a subprocess, needed there to isolate
    MediaPipe's per-frame CPU cost from the GIL), this runs as a plain
    thread in the main process: there's no CPU-bound inference here at
    all, just cap.read() + a queue + disk write via _video_writer_loop
    (shared verbatim with the primary camera, just pointed at
    config.CAM2_VIDEO_STEM/CAM2_FRAMES_CSV instead) — exactly the
    lightweight "capture path never blocked by disk I/O" pattern that
    function already implements, so there's nothing to duplicate here.

    Takes session_start_wall (not its own independent clock) — the same
    wall-clock reference the primary camera anchors to (see this module's
    docstring for why time.time(), not perf_counter()) — so this camera's
    frames.csv timestamps land on the exact same timeline session.py's
    _extract_trial_videos() already crops every stream against. No
    separate sync step needed for this second stream as a result.

    stop_event is state.stop_event directly (a threading.Event, not the
    mp.Event camera_process_fn's subprocess needs) — call from run.py's
    main() as a plain threading.Thread, not multiprocessing.Process.
    """
    import cv2 as _cv2
    from ..core import config

    cap = _cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f'[CAMERA2] could not open camera index {camera_index} — camera2 recording disabled this session')
        return

    # See camera_process_fn's identical block for why these are requested
    # here (before any cap.read()) and why the actually-negotiated values
    # are read back and logged rather than trusted.
    cap.set(_cv2.CAP_PROP_FRAME_WIDTH, config.CAM2_CAPTURE_WIDTH)
    cap.set(_cv2.CAP_PROP_FRAME_HEIGHT, config.CAM2_CAPTURE_HEIGHT)
    cap.set(_cv2.CAP_PROP_FPS, config.CAM2_CAPTURE_FPS_REQUEST)
    actual_w = cap.get(_cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(_cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(_cv2.CAP_PROP_FPS)
    print(f'[CAMERA2] requested {config.CAM2_CAPTURE_WIDTH}x{config.CAM2_CAPTURE_HEIGHT}'
          f'@{config.CAM2_CAPTURE_FPS_REQUEST}fps — camera reports {actual_w:.0f}x{actual_h:.0f}'
          f'@{actual_fps:.1f}fps (driver-reported FPS is nominal, not measured — camera2_frames.csv '
          f'is what confirms the real achieved rate)')

    video_queue: "queue.Queue" = queue.Queue(maxsize=config.CAM2_VIDEO_QUEUE_MAXSIZE)
    writer_thread = threading.Thread(
        target=_video_writer_loop,
        args=(video_queue, session_dir, config.CAM_VIDEO_CODEC, config.CAM_VIDEO_EXT,
              config.CAM2_CAPTURE_FPS_REQUEST, config.CAM_FLUSH_EVERY_N,
              config.CAM2_VIDEO_STEM, config.CAM2_FRAMES_CSV),
        daemon=True,
    )
    writer_thread.start()

    dropped = 0
    while not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            continue
        ts = time.time() - session_start_wall
        try:
            video_queue.put_nowait((frame.copy(), ts))
        except queue.Full:
            dropped += 1

    video_queue.put(None)
    writer_thread.join(timeout=config.CAM2_VIDEO_DRAIN_TIMEOUT_SEC)
    if writer_thread.is_alive():
        print('[CAMERA2] video writer thread did not finish flushing in time — '
              'camera2_raw file may be missing trailing frames')
    if dropped:
        print(f'[CAMERA2] {dropped} frames dropped to backpressure this session')

    cap.release()