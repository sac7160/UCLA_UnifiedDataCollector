"""
data_collector/core/state.py
────────────────────────────────────────────────────────────────────────────
Every piece of mutable state that more than one module needs to read or
write lives here, as plain module-level attributes. Other modules do

    from data_collector import state
    ...
    state.some_var = new_value      # write
    x = state.some_var              # read

instead of Python's `global` keyword, because `global` only ever refers to
the *current* module's namespace — it can't reach into another module. This
is the one file that makes splitting a script this stateful into multiple
files actually work; every other module in this package imports this one
(never the other way around), so there's no circular-import risk.

Grouped by the subsystem that owns each block below, matching the module
that primarily writes to it (though many are read from several places).
"""

from __future__ import annotations

import queue
import threading
from collections import deque

from . import config

stop_event = threading.Event()

# ─── Locks ────────────────────────────────────────────────────────────────────
file_lock     = threading.Lock()   # guards session file handles / _sync dict
trial_lock    = threading.Lock()   # guards _trial_active / _trial_buffers
pending_lock  = threading.Lock()   # guards _pending_starts
rolling_lock  = threading.Lock()   # guards _imu_rolling / _watch_audio_rolling

# ─── Session / file state (session.py owns writes) ───────────────────────────
session_dir = None                  # Path | None
session_start: float | None = None
session_start_wall: float | None = None   # time.time() at session start — see camera.py's docstring for why
                                            # this (not session_start) is what crosses the process boundary
                                            # into the camera subprocess

watch_wf = None                     # wave.Wave_write | None — owned by watch_network.py's audio worker
mic_wf   = None                     # wave.Wave_write | None — owned by touch_detection.py's mic wav writer
imu_fp = None
imu_writer = None
imu_flush_n = 0

cam_fp = None
cam_writer = None
cam_flush_n = 0

traj_fp = None
traj_writer = None
traj_flush_n = 0

events_fp = None
events_writer = None
event_log: list = []   # (ts, event) for the whole session — see trial.write_event()/write_event_at();
                        # process_trial() filters this down to each trial's window and saves it
                        # alongside imu.csv/fingertip_imu.csv, on the same time_aligned convention
space_down = False   # guards OS key-repeat while spacebar is physically held

# ─── Instructor window's live terminal/log panel (utils.install_stdout_tee) ──
log_lock = threading.Lock()
log_lines = deque(maxlen=500)   # rolling buffer of recent stdout lines
log_seq = 0                     # monotonically increasing — GUI compares against its own
                                 # last-seen value to know how many new lines arrived since the last tick

watch_audio_frames_fp = None
watch_audio_frames_writer = None
watch_audio_session_samples = 0

# ─── Periodic "still receiving watch data" heartbeat (watch_network.py) ──────
# Incremented on every real packet, printed+reset every HEARTBEAT_INTERVAL_SEC
# by heartbeat_thread_fn() — this is the one thing that prints during normal
# operation, so it stays a single line every few seconds rather than a line
# per packet (which is what --verbose is for, off by default).
heartbeat_lock = threading.Lock()
heartbeat_audio_frames = 0
heartbeat_imu_acc = 0
heartbeat_imu_gyro = 0

watch_audio_offset: float | None = None
mic_offset:         float | None = None
imu_offset:         float | None = None
cam_offset:         float | None = None
sync: dict = {}

watch_rms: float = 0.0
mic_rms:   float = 0.0

verbose = False   # set from --verbose

# ─── Trial buffering / two-tier trial boundaries (trial.py owns writes) ──────
trial_active = False                # True while a REC session is buffering
trial_start_offset: float | None = None
trial_buffers = {
    # watch IMU / watch audio arrive in network batches with inherent
    # latency, so they're handled separately via the rolling buffers below
    # instead of being gated directly by REC state.
    'fingertip': [],
    'trajectory': [],
    'mic':       [],
}
trial_queue: "queue.Queue" = queue.Queue()   # carries (start, end, snapshot, trigger, label)
mic_sr_runtime: int = config.MIC_SR          # updated in main() to the actual mic sample rate in use

rec_active: bool = False
audio_touch_start: float | None = None   # set while a touch is in progress during REC — see touch_detection.py
current_label: str = ''         # kept live by the instructor window's class-picker dropdown

pending_starts: list = []

imu_rolling:         "deque" = deque()   # (ts, sensor, v1, v2, v3, watch_ts_ms)
watch_audio_rolling: "deque" = deque()   # (ts, raw_bytes, watch_ts_ms)

trial_dataset_root = None   # Path — set from --dataset-root in main()
trial_margin: float = 0.1   # trimmed inward from both REC start/stop, on every trial (human reaction delay to the spacebar press)

# ─── Surface mic filter state (touch_detection.py owns) ──────────────────────
mic_target_ch = 0   # 0-indexed, set from --mic-channel

# ─── Display ring buffers (all modules read; touch_detection.py /
# watch_network.py / camera.py push into them) ────────────────────────────────
# NOTE: these hold ScrollingWaveform/ScrollingSpectrogram/ScrollingIMU
# instances (see data_collector/gui/display_buffers.py) — left untyped here and
# created in data_collector.py's main() instead of imported directly, so
# that core/ (this package) never has to import anything from gui/. core/
# is meant to be the one package everything else depends on, never the
# other way around.
disp_surface_wave = None
disp_surface_spec = None
disp_watch_wave   = None
disp_watch_spec   = None
disp_watch_acc  = None
disp_watch_gyro = None
disp_finger_acc  = None
disp_finger_gyro = None
disp_trajectory  = None   # TrajectoryTrail — index-fingertip 2D trail, see gui/display_buffers.py
display_finger = 'index'   # which finger's fingertip IMU is shown; set from --finger / GUI

# ─── Touch detection (touch_detection.py owns) ───────────────────────────────
touch_band_low_hz  = config.MATERIAL_PRESETS['wood'][0]
touch_band_high_hz = config.MATERIAL_PRESETS['wood'][1]
current_material   = 'wood'

mic_band_sos = None
mic_band_zi  = None
envelope = 0.0
noise_floor: float | None = None    # fixed by calibration; unchanged until the next recalibration
noise_floor_db_abs = -100.0         # same value as noise_floor, in dB, purely for the status readout
touch_metric_db = -60.0

is_calibrating = True
calibration_start_time: float | None = None
calibration_samples: list = []

touch_on_threshold_db  = config.TOUCH_ON_THRESHOLD_DB
touch_off_threshold_db = config.TOUCH_OFF_THRESHOLD_DB

touch_min_on_sec  = config.TOUCH_MIN_ON_MS_DEFAULT / 1000.0
touch_min_off_sec = config.TOUCH_MIN_OFF_MS_DEFAULT / 1000.0
touch_on_state = False
touch_candidate_on_time  = 0.0
touch_candidate_off_time = 0.0
touch_median_buf: "deque | None" = None

material_change_queue: "queue.Queue" = queue.Queue(maxsize=1)   # (name, band_low, band_high) requests

# ─── Inter-thread queues ──────────────────────────────────────────────────────
audio_process_queue: "queue.Queue" = queue.Queue()   # (raw, amplified, frames, arrival_pc) -> touch_detection worker
mic_wav_queue:       "queue.Queue" = queue.Queue()   # raw int16 PCM bytes -> mic wav writer
watch_audio_queue:   "queue.Queue" = queue.Queue()   # (payload, arrival_pc) or ('__RTEND__', ts) -> watch audio worker
watch_imu_queue:     "queue.Queue" = queue.Queue()   # (payload, sensor, arrival_pc) -> watch IMU worker

# ─── Instructor -> experimenter shared text ──────────────────────────────────
current_stimulus: str = ''   # what the experimenter should write right now