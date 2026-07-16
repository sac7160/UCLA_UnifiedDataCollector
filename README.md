# UnifiedDataCollector

A system for simultaneously collecting Galaxy Watch8 IMU/audio, a surface contact microphone, and a camera-based fingertip virtual IMU. Built as a multimodal data acquisition pipeline for on-surface handwriting tracking/recognition research (WatchTouch).

## Collected Streams

| Stream                | Source               | Notes                                                |
| --------------------- | -------------------- | ---------------------------------------------------- |
| Watch IMU (acc/gyro)  | Galaxy Watch8, TCP   | Includes a timestamp based on the watch's own clock  |
| Watch audio           | Galaxy Watch8, TCP   | Each frame (40ms) includes its own capture timestamp |
| Surface mic           | USB audio interface  | Surface-contact microphone (ASIN B0F5GGGSNG)         |
| Fingertip virtual IMU | PC camera, MediaPipe | Accel/gyro for all 5 fingertips                      |

---

## Project Structure

```
UnifiedDataCollector/
├── app/                          # Wear OS app (build onto the watch)
├── gradle/, build.gradle.kts, gradlew ...  # Android build files
└── run/                          # PC-side collection scripts
    ├── unified_collector_final.py    # main entry point
    ├── fingertip_imu_multi.py        # camera-based fingertip virtual IMU module
    ├── plot_imu.py                   # watch IMU vs. fingertip IMU comparison plot
    ├── data/                         # session-level raw recordings (auto-created)
    └── dataset/                      # trial-level cropped recordings (auto-created)
        ├── <label>/
        │   ├── trial_001/
        │   ├── trial_002/
        │   └── ...
        └── metadata.csv
```

---

## Setup

### 1. Build the watch app

Open the `UnifiedDataCollector` project (the `app` module) in Android Studio and **build & install it directly onto the Galaxy Watch**. The watch and the PC must be on the **same WiFi network**.

### 2. PC environment

```bash
pip install sounddevice numpy scipy opencv-python mediapipe pynput
```

### 3. Audio interface

Connect the surface mic to the USB audio interface, then check the device/channel:

```bash
python unified_collector_final.py --list-devices
```

---

## Data Collection Procedure

1. From the `run/` folder, start the collector (use `--label` to name the stroke/letter class being repeated this session):
    ```bash
    python unified_collector_final.py --label line_horizontal
    ```
2. Launch the watch app, **enter the PC's IP address, and press Start** to begin streaming. (The console prints the PC's IP as `[NET] Watch TCP: <IP>:50005`.)
3. Holding the **spacebar down with the opposite hand** marks one trial (a single writing motion).
4. To stop, press `Ctrl+C` (or `q` if the camera preview window is open).

### Key CLI options

| Option                          | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `--label <name>`                | Class label being repeated in this session                   |
| `--mic-device`, `--mic-channel` | Specify the audio interface device/channel                   |
| `--dataset-root`                | Trial dataset output path (default: `dataset/`)              |
| `--trial-margin`                | Margin trimmed at touch_down/up boundaries (default: 0.1s)   |
| `--no-camera` / `--show-camera` | Disable the camera (fingertip IMU) / show the preview window |
| `--camera-pitch-deg`            | Camera tilt angle (for gravity compensation)                 |

---

## Output Data Layout

**Session level** (`data/session_YYYYMMDD_HHMMSS_<label>/`) — the full raw recording

```
watch_audio.wav   surface_mic.wav   imu.csv   fingertip_imu.csv   events.csv   sync.json
```

**Trial level** (`dataset/<label>/trial_XXX/`) — training-ready data cropped in real time per spacebar-marked segment

```
watch_audio.wav   surface_mic.wav   imu.csv   fingertip_imu.csv
```

All trial records accumulate in `dataset/metadata.csv`.

---

## Synchronization

- All streams are aligned to the PC's common reference clock (`time.perf_counter()`).
- Watch IMU/audio also carry the **watch's own clock timestamp** (`watch_ts_ms`), which is mapped to PC time via the RTBGN/RTEND reference — alignment reflects true capture time regardless of network delay or arrival order.
- The watch and PC communicate over a **persistent TCP connection** established at session start, with automatic reconnection if the connection drops.

---

## Verification & Diagnostics

- A `[QUALITY]` log is printed at the end of each session. Sessions with an unstable watch connection will show a warning like the one below — **re-collecting is recommended** in that case:
    ```
    [QUALITY] WARNING: watch connection may have stalled during this session ...
    ```
- Use `plot_imu.py` to visually compare a trial's watch IMU against its fingertip virtual IMU for a quick data-quality check:
    ```bash
    python plot_imu.py --imu <trial_dir>/imu.csv --fingertip <trial_dir>/fingertip_imu.csv --finger index
    ```

---

## Known Limitations

- The fingertip virtual IMU's frame rate is capped around 22-23Hz, limited by the webcam hardware (30fps) and MediaPipe inference speed.
- Surface mic gain depends on the audio interface's hardware preamp — re-check signal level whenever the surface material changes.
- Spacebar-based touch_down/up marking has human reaction delay (~100-300ms), so it's not precise enough for trajectory ground truth — use it only for cropping trial boundaries for stroke/letter classification.
