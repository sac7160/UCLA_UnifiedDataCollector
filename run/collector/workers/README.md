# WatchTouch.workers

Everything that captures, detects, or saves sensor data.

Depends on `watchtouch.core` only — never on `watchtouch.gui`.

| File                 | What's in it                                                                                                |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| `session.py`         | session file lifecycle (open/close) + post-hoc trial recalibration                                          |
| `writers.py`         | `write_watch_audio` / `write_imu` / `write_fingertip_imu` — per-stream session-file + display-buffer writes |
| `touch_detection.py` | mic callback + band-pass/calibration/threshold/debounce touch detection                                     |
| `watch_network.py`   | watch TCP listener + packet parsing                                                                         |
| `camera.py`          | MediaPipe fingertip-IMU process + bridge thread                                                             |
| `trial.py`           | REC toggle, two-tier trial boundaries, trial cropping/saving                                                |
