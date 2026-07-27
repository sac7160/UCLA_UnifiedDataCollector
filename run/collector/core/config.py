"""
wristpad/core/config.py
────────────────────────────────────────────────────────────────────────────
Pure constants — no shared mutable state lives here (see state.py for that).
Safe to import from anywhere without creating circular-import problems.
"""

from pathlib import Path

# ─── Watch TCP ────────────────────────────────────────────────────────────────
WATCH_HOST       = '0.0.0.0'
WATCH_PORT       = 50005
WATCH_AUDIO_SR   = 48000
WATCH_FRAME_SIZE = WATCH_AUDIO_SR // 25   # 1920 samples
WATCH_BUF_SIZE   = WATCH_FRAME_SIZE * 2   # 3840 bytes

# The watch timestamps an audio frame only once it's fully buffered, so
# watch_ts_ms is systematically late relative to true capture time.
# Corrected once, at trial-crop time, after the RTBGN-based watch-clock ->
# PC-time mapping is known.
WATCH_AUDIO_LATENCY_SEC = 0.04#0.07#0.045

# ─── Surface mic ──────────────────────────────────────────────────────────────
MIC_SR         = 192000
MIC_CHANNELS   = 4
MIC_TARGET_CH_DEFAULT = 1
MIC_BLOCK_SIZE = 512
MIC_GAIN_DEFAULT = 1.0

# ─── Camera / fingertip IMU ───────────────────────────────────────────────────
CAM_SMOOTHING_WINDOW = 3
CAM_EMA_ALPHA        = 0.2
CAM_FLUSH_EVERY_N    = 10
FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']

# ─── Index-finger trajectory (see trajectory_calibration.py) ──────────────────
TRAJ_TRAIL_MAXLEN = 150   # points kept for the instructor window's live trail plot

# ─── Camera raw video recording ────────────────────────────────────────────
# MJPG is deliberately chosen over an inter-frame codec (mp4v/avc1): every
# frame is independently JPEG-encoded, so there's no encoder lookahead/
# backlog risk under load, MJPG/.avi support is close to universal across
# OpenCV builds (H.264 availability depends on how the local OpenCV was
# compiled), and — the property session.py's _extract_trial_videos() relies
# on — an all-intra file has no keyframe intervals, so `ffmpeg -c copy` can
# cut it at any frame boundary losslessly. Fourcc and extension are paired;
# change both together.
CAM_VIDEO_CODEC             = 'MJPG'
CAM_VIDEO_EXT                = '.avi'
CAM_VIDEO_FALLBACK_FPS       = 30.0   # used only if cap.get(cv2.CAP_PROP_FPS) is 0/invalid — container
                                       # metadata only, camera_frames.csv is the source of truth for
                                       # actual per-frame timing (cap.read() isn't perfectly periodic)
CAM_VIDEO_QUEUE_MAXSIZE      = 60     # in-process queue.Queue depth inside the camera subprocess (~2s @30fps)
CAM_VIDEO_DRAIN_TIMEOUT_SEC  = 3.0    # camera_process_fn's own deadline for flushing the writer backlog at
                                       # shutdown — keep run.py's cam_proc.join(timeout=...) comfortably above this
CAM_VIDEO_FPS_CORRECTION_THRESHOLD = 0.05   # relative difference (vs CAM_VIDEO_FALLBACK_FPS) above which
                                             # session.py's _fix_video_framerate() remuxes camera_raw.avi with
                                             # the true measured fps from camera_frames.csv's own timestamps —
                                             # the live recording fps is only ever a provisional guess (see
                                             # camera.py), this is what makes it correct after the fact

# ─── Session / dataset ────────────────────────────────────────────────────────
DATA_ROOT      = Path('data')
SESSION_PREFIX = 'session'

# ─── Trial buffering ──────────────────────────────────────────────────────────
ROLLING_RETENTION_SEC = 30.0
IMU_GRACE_SEC         = 0.5
WATCH_AUDIO_GRACE_SEC = 0.5

# ─── Touch detection ──────────────────────────────────────────────────────────
# Material -> (band_low_hz, band_high_hz) for the touch-detection band-pass.
# NOTE: all four currently point at the same 3000-6000Hz range (from the
# acrylic measurements) — this is presumably deliberate for the current
# test setup, but means the material buttons don't actually change
# anything but the label/metadata right now. If wood/paper/fabric turn out
# to need a different band later, only this dict needs updating.
MATERIAL_PRESETS = {
    'wood':    (3000.0, 6000.0),
    'paper':   (3000.0, 6000.0),
    'fabric':  (3000.0, 6000.0),
    'acrylic': (3000.0, 6000.0),
}

ENV_ATTACK_TAU_SEC  = 0.005
ENV_RELEASE_TAU_SEC = 0.08
CALIBRATION_DURATION_SEC = 1.5   # how long to listen quietly before fixing the floor
# The ONLY place the touch on/off decision thresholds are set — the
# instructor window's threshold/hysteresis spinboxes are display-only
# (disabled) in this calibrated-floor design, since there's no live
# "drag the slider and see the effect" tuning anymore. To change the
# thresholds, edit these two constants and restart (or recalibrate, which
# does not touch these — only the floor).
TOUCH_ON_THRESHOLD_DB  = 8.0
TOUCH_OFF_THRESHOLD_DB = 5.0

# Material -> (on_threshold_db, off_threshold_db). Different surfaces pick
# up finger contact at different loudness relative to their own ambient
# noise, so a threshold tuned for wood can be way too sensitive (or not
# sensitive enough) on acrylic. Applied on every material switch (button
# click or startup), same as MATERIAL_PRESETS' band above — edit this
# dict and reselect the material (or restart) to pick up new values, no
# code changes needed elsewhere. A material missing from this dict falls
# back to TOUCH_ON_THRESHOLD_DB / TOUCH_OFF_THRESHOLD_DB above.
MATERIAL_THRESHOLDS = {
    'wood':    (25.0, 22.0),
    'paper':   (8.0, 5.0),
    'fabric':  (8.0, 5.0),
    'acrylic': (8.0, 5.0),
}

TOUCH_MIN_ON_MS_DEFAULT  = 30.0
TOUCH_MIN_OFF_MS_DEFAULT = 100.0
TOUCH_MEDIAN_WINDOW_DEFAULT = 3

# ─── GUI ──────────────────────────────────────────────────────────────────────
AXIS_COLORS = {'x': '#d62728', 'y': '#2ca02c', 'z': '#1f77b4'}