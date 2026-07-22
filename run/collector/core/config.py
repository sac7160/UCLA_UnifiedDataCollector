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
WATCH_AUDIO_LATENCY_SEC = 0.07#0.045

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

TOUCH_MIN_ON_MS_DEFAULT  = 30.0
TOUCH_MIN_OFF_MS_DEFAULT = 100.0
TOUCH_MEDIAN_WINDOW_DEFAULT = 3

# ─── GUI ──────────────────────────────────────────────────────────────────────
AXIS_COLORS = {'x': '#d62728', 'y': '#2ca02c', 'z': '#1f77b4'}
