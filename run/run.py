"""
run.py
────────────────────────────────────────────────────────────────────────────
WristPad data-collection app — entry point. All the actual logic lives in
the collector/ package next to this file, split into three subpackages
(dependency direction: gui -> workers -> core, one-way):

    collector/core/
        config.py              constants
        state.py               shared mutable state (see its own docstring —
                                this is what makes the multi-file split work
                                with code that leans on `global`)
        utils.py               offset()/log() and the rolling-buffer cutoff helper
    collector/workers/
        session.py             session file lifecycle + post-hoc trial recalibration
        writers.py              write_watch_audio / write_imu / write_fingertip_imu / write_trajectory
        touch_detection.py     mic callback + band-pass/calibration/threshold/debounce
        watch_network.py        watch TCP listener + packet parsing
        camera.py                MediaPipe fingertip-IMU + trajectory process + bridge thread
        calibration.py           interactive index-fingertip trajectory calibration (subprocess)
        trial.py                 REC toggle, two-tier trial boundaries, trial cropping/saving
    collector/gui/
        display_buffers.py       ScrollingWaveform/Spectrogram/IMU/TrajectoryTrail
        instructor_window.py     InstructorWindow
        experimenter_window.py   ExperimenterWindow

This file itself only does: argparse, running the pre-recording trajectory
calibration prompt, opening the mic stream, starting every thread/process in
the right order, building the two windows, and tearing everything down
cleanly on exit.

Usage:
    python run.py
    python run.py --mic-device 1 --mic-channel 1
    python run.py --dataset-root dataset/ --list-devices
    python run.py --skip-calibration   # reuse calibration.json without the interactive prompt
"""

import argparse
import multiprocessing as mp
import signal
import sys
import threading
from collections import deque
from pathlib import Path

import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pynput import keyboard

from collector.core import config, state
from collector.core.utils import install_stdout_tee
from collector.gui.display_buffers import ScrollingWaveform, ScrollingSpectrogram, ScrollingIMU, TrajectoryTrail
from collector.workers.calibration import get_calibration
from collector.workers.session import start_session, close_session
from collector.workers.touch_detection import (
    mic_callback, mic_wav_writer_fn, audio_worker_fn, rebuild_touch_band_filter,
)
from collector.workers.watch_network import (
    net_thread_fn, watch_audio_worker_fn, watch_imu_worker_fn, heartbeat_thread_fn,
)
from collector.workers.camera import camera_process_fn, camera_bridge_thread_fn
from collector.workers.trial import trial_worker_fn, on_key_press, on_key_release, write_event
from collector.gui.instructor_window import InstructorWindow
from collector.gui.experimenter_window import ExperimenterWindow


def main():
    parser = argparse.ArgumentParser(description='WristPad experiment collector')
    parser.add_argument('--mic-device', type=int, default=None)
    parser.add_argument('--mic-channel', type=int, default=1)
    parser.add_argument('--mic-channels', type=int, default=config.MIC_CHANNELS,
                         help=f'total input channel count to open on the mic device '
                              f'(default {config.MIC_CHANNELS}, sized for a 4-channel interface like the '
                              f'TASCAM US-4x4; set to your device\'s actual channel count, e.g. 2, '
                              f'if PortAudio reports "Invalid number of channels")')
    parser.add_argument('--mic-sr', type=int, default=config.MIC_SR)
    parser.add_argument('--mic-gain', type=float, default=1.0)
    parser.add_argument('--watch-port', type=int, default=config.WATCH_PORT)
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--camera-pitch-deg', type=float, default=None)
    parser.add_argument('--camera-roll-deg', type=float, default=0.0)
    parser.add_argument('--no-camera', action='store_true')
    parser.add_argument('--skip-calibration', action='store_true',
                         help="skip the interactive pre-recording trajectory calibration prompt "
                              "(just silently reuses calibration.json if it exists, else records "
                              "uncalibrated trajectory values)")
    parser.add_argument('--finger', choices=config.FINGER_NAMES, default='index')
    parser.add_argument('--dataset-root', type=Path, default=Path('dataset'))
    parser.add_argument('--session-label', default='',
                         help='optional suffix for the session folder name (e.g. a participant ID) '
                              '— data/session_YYYYMMDD_HHMMSS_<this>/. Does NOT set what gets '
                              'written; that\'s chosen live from the instructor window\'s class '
                              'dropdown, per trial.')
    parser.add_argument('--trial-margin', type=float, default=0.1)
    parser.add_argument('--material', choices=list(config.MATERIAL_PRESETS), default='wood')
    parser.add_argument('--touch-min-on-ms', type=float, default=config.TOUCH_MIN_ON_MS_DEFAULT)
    parser.add_argument('--touch-min-off-ms', type=float, default=config.TOUCH_MIN_OFF_MS_DEFAULT)
    parser.add_argument('--touch-median-window', type=int, default=config.TOUCH_MEDIAN_WINDOW_DEFAULT)
    parser.add_argument('--window-sec', type=float, default=2.0)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--display-hz', type=int, default=8000)
    parser.add_argument('--opengl', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--list-devices', action='store_true')
    args = parser.parse_args()
    state.verbose = args.verbose

    if args.list_devices:
        print('\n=== Audio Devices ===')
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                print(f'  [{i:2d}] {d["name"]} (in={d["max_input_channels"]}, sr={int(d["default_samplerate"])})')
        return

    install_stdout_tee()   # from here on, every print()/log() also reaches the instructor window's log panel

    app = pg.mkQApp('WristPad Experiment Collector')

    def _handle_sigint(signum, frame):
        print('\n[RUN] Ctrl+C received — shutting down...')
        app.quit()
    signal.signal(signal.SIGINT, _handle_sigint)

    state.display_finger = args.finger
    mic_target_ch_1indexed = args.mic_channel

    mic_device = args.mic_device
    if mic_device is None:
        for i, d in enumerate(sd.query_devices()):
            name = d['name'].lower()
            if ('tascam' in name or 'us-4x4' in name) and d['max_input_channels'] > 0:
                mic_device = i
                break
            if ('focusrite' in name or 'scarlett' in name) and d['max_input_channels'] > 0:
                mic_device = i
                if mic_target_ch_1indexed == 1:
                    mic_target_ch_1indexed = 2
                break
    if mic_device is None:
        print('[MIC] No audio interface found. Specify one with --mic-device N.')
        sys.exit(1)

    device_max_ch = sd.query_devices(mic_device)['max_input_channels']
    if args.mic_channels > device_max_ch:
        print(f'[MIC] --mic-channels={args.mic_channels} exceeds device [{mic_device}]\'s '
              f'max_input_channels={device_max_ch}. Pass --mic-channels {device_max_ch} '
              f'(see --list-devices for other options).')
        sys.exit(1)
    if mic_target_ch_1indexed > args.mic_channels:
        print(f'[MIC] --mic-channel={mic_target_ch_1indexed} is out of range for '
              f'--mic-channels={args.mic_channels}.')
        sys.exit(1)
    state.mic_target_ch = mic_target_ch_1indexed - 1

    # Trajectory calibration happens before the session starts (and before
    # the camera subprocess opens the device), so calibration time doesn't
    # count toward the session clock and the two never fight over the
    # camera — see collector/workers/calibration.py.
    camera_calibration = None
    if not args.no_camera:
        camera_calibration = get_calibration(args.camera_index, args.skip_calibration)

    config.DATA_ROOT.mkdir(parents=True, exist_ok=True)
    start_session(label=args.session_label)

    state.mic_sr_runtime = args.mic_sr
    state.trial_dataset_root = args.dataset_root
    state.trial_margin = args.trial_margin
    state.trial_dataset_root.mkdir(parents=True, exist_ok=True)

    state.touch_median_buf = deque(maxlen=max(1, args.touch_median_window))
    state.touch_min_on_sec = args.touch_min_on_ms / 1000.0
    state.touch_min_off_sec = args.touch_min_off_ms / 1000.0
    state.current_material = args.material
    band_low, band_high = config.MATERIAL_PRESETS[args.material]
    rebuild_touch_band_filter(args.mic_sr, band_low, band_high)
    write_event(f'material:{args.material}')

    surface_decimate = max(1, args.mic_sr // args.display_hz)
    state.disp_surface_wave = ScrollingWaveform(args.mic_sr, args.window_sec, decimate=surface_decimate)
    state.disp_surface_spec = ScrollingSpectrogram(args.mic_sr, n_fft=1024, hop_sec=0.02,
                                                     max_cols=int(args.window_sec / 0.02), freq_max=10000.0)
    watch_decimate = max(1, config.WATCH_AUDIO_SR // args.display_hz)
    state.disp_watch_wave = ScrollingWaveform(config.WATCH_AUDIO_SR, args.window_sec, decimate=watch_decimate)
    state.disp_watch_spec = ScrollingSpectrogram(config.WATCH_AUDIO_SR, n_fft=1024, hop_sec=0.03,
                                                   max_cols=int(args.window_sec / 0.03), freq_max=6000.0)
    # These four don't depend on any CLI arg (fixed window/rate), but are
    # still created here rather than in core/state.py so that core/ never
    # has to import from gui/ — see core/state.py's docstring.
    state.disp_watch_acc   = ScrollingIMU(window_sec=5.0, expected_hz=100.0)
    state.disp_watch_gyro  = ScrollingIMU(window_sec=5.0, expected_hz=100.0)
    state.disp_finger_acc  = ScrollingIMU(window_sec=5.0, expected_hz=30.0)
    state.disp_finger_gyro = ScrollingIMU(window_sec=5.0, expected_hz=30.0)
    state.disp_trajectory  = TrajectoryTrail(maxlen=config.TRAJ_TRAIL_MAXLEN)

    mic_stream = sd.InputStream(
        device=mic_device, channels=args.mic_channels, samplerate=args.mic_sr,
        blocksize=config.MIC_BLOCK_SIZE, dtype='float32', callback=mic_callback,
    )
    mic_stream.start()

    net_t = threading.Thread(target=net_thread_fn, args=(args.watch_port,), daemon=True)
    net_t.start()

    watch_audio_worker_t = threading.Thread(target=watch_audio_worker_fn, daemon=True)
    watch_audio_worker_t.start()
    watch_imu_worker_t = threading.Thread(target=watch_imu_worker_fn, daemon=True)
    watch_imu_worker_t.start()
    heartbeat_t = threading.Thread(target=heartbeat_thread_fn, daemon=True)
    heartbeat_t.start()

    cam_proc = cam_bridge_t = record_queue = cam_stop_flag = None
    if not args.no_camera:
        record_queue = mp.Queue(maxsize=8)
        cam_stop_flag = mp.Event()
        cam_proc = mp.Process(
            target=camera_process_fn,
            args=(args.camera_index, args.camera_pitch_deg, args.camera_roll_deg,
                  state.session_start_wall, record_queue, cam_stop_flag, camera_calibration),
            daemon=True,
        )
        cam_proc.start()
        cam_bridge_t = threading.Thread(target=camera_bridge_thread_fn, args=(record_queue,), daemon=True)
        cam_bridge_t.start()

    trial_worker_t = threading.Thread(target=trial_worker_fn, daemon=True)
    trial_worker_t.start()

    audio_worker_t = threading.Thread(target=audio_worker_fn, daemon=True)
    audio_worker_t.start()

    mic_wav_writer_t = threading.Thread(target=mic_wav_writer_fn, daemon=True)
    mic_wav_writer_t.start()

    key_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    try:
        key_listener.start()
    except Exception:
        pass

    instructor = InstructorWindow(args.window_sec, has_camera=not args.no_camera,
                                   use_opengl=args.opengl)
    # Per-trial labeling now comes from the instructor window's class-picker
    # dropdown (see InstructorWindow._on_class_changed), not a CLI flag or
    # text field — nothing to wire up here. --session-label still exists,
    # but only names the session folder now (e.g. a participant ID).

    experimenter = ExperimenterWindow()

    # NOTE: the experimenter window is a normal, maximized window rather
    # than fullscreen. On macOS a fullscreen Qt window switches to its own
    # Space and can steal input focus from other windows — this was also
    # why the instructor window's buttons stopped responding in an earlier
    # version.
    screens = QtGui.QGuiApplication.screens()
    if len(screens) >= 2:
        instructor.setGeometry(screens[0].geometry())
        experimenter.setGeometry(screens[1].geometry())
        experimenter.showMaximized()
    else:
        experimenter.move(instructor.x() + 50, instructor.y() + 50)
    instructor.show()
    experimenter.show()

    timer = QtCore.QTimer()
    def _tick():
        instructor.update()
        experimenter.update()
    timer.timeout.connect(_tick)
    timer.start(max(1, int(1000 / args.fps)))

    print('[RUN] Instructor + experimenter windows open. Close either window or Ctrl+C to stop.')

    shutdown_done = False

    def _shutdown():
        nonlocal shutdown_done
        if shutdown_done:
            return
        shutdown_done = True
        state.stop_event.set()
        if cam_stop_flag:
            cam_stop_flag.set()
        mic_stream.stop(); mic_stream.close()   # stops the callback, so no new items get queued after this
        mic_wav_writer_t.join(timeout=2.0)
        net_t.join(timeout=2.0)   # stops before its consumers, so no new items get queued after this
        watch_audio_worker_t.join(timeout=2.0)
        watch_imu_worker_t.join(timeout=2.0)
        heartbeat_t.join(timeout=2.0)
        if cam_proc:
            cam_proc.join(timeout=2.0)
            if cam_proc.is_alive():
                cam_proc.terminate()
        if cam_bridge_t:
            cam_bridge_t.join(timeout=2.0)
        try:
            key_listener.stop()
        except Exception:
            pass
        trial_worker_t.join(timeout=2.0)
        audio_worker_t.join(timeout=2.0)
        close_session()

    app.aboutToQuit.connect(_shutdown)

    try:
        if hasattr(pg, 'exec'):
            pg.exec()
        else:
            app.exec_()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown()


if __name__ == '__main__':
    main()