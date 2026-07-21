"""
index_trajectory_viewer.py
────────────────────────────────────────────────────────────────────────────
Standalone demo: tracks the index fingertip's 2D (pixel/image-plane) and 3D
(mm, world-frame) trajectory using MediaPipe Hands, via the same tracker
unified_collector_final.py uses (fingertip_imu_multi.MultiFingertipIMUTracker).

Calibration and trajectory computation live in trajectory_calibration.py,
shared with unified_collector_final.py so both compute/log identical columns
(see that module's docstring for what calibration does).

Draws a live fading trail on the camera preview and logs both positions to a
CSV file, one row per camera frame.

Trajectory CSVs are saved under trajectory_sessions/ (created alongside this
script if it doesn't exist yet), not directly in run/. Calibration itself is
shared and saved to run/calibration.json.

Usage:
    python index_trajectory_viewer.py
    python index_trajectory_viewer.py --camera-index 1 --output my_traj.csv
    python index_trajectory_viewer.py --trail-length 90 --no-skeleton
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

from fingertip_imu_multi import MultiFingertipIMUTracker
from trajectory_calibration import (
    compute_trajectory,
    load_calibration,
    run_calibration,
    trajectory_csv_row,
    TRAJECTORY_CSV_HEADER,
)

TRAIL_COLOR = (60, 220, 255)   # BGR
FLUSH_EVERY_N_FRAMES = 30
TRAJECTORY_DIR = Path(__file__).resolve().parent / 'trajectory_sessions'
WINDOW_NAME = 'Index Fingertip Trajectory'


def _draw_readout(frame, detected: bool, x_px, y_px, pos_x, pos_y, pos_z,
                   local_vec, global_xy, frame_xy, height_mm, calibration: dict | None):
    """Live numeric overlay so the 2D/3D/global coordinates can be
    sanity-checked by eye in real time instead of only reading them back from
    the CSV afterward."""
    if calibration is None:
        calib_line = 'calib   : not set — press c'
    elif calibration.get('mm_per_pixel') is None:
        calib_line = f'calib   : local scale={calibration["scale_factor"]:.3f}, no mic origin (press c to redo)'
    else:
        calib_line = (f'calib   : local scale={calibration["scale_factor"]:.3f}, '
                       f'mic mm/px={calibration["mm_per_pixel"]:.3f} (press c to redo)')

    if local_vec is not None:
        label = 'calibrated' if calibration is not None else 'raw'
        local_line = (f'local mm({label}): x={local_vec[0]:7.1f}  '
                       f'y={local_vec[1]:7.1f}  z={local_vec[2]:7.1f}')
    else:
        local_line = 'local mm: --'

    if global_xy is not None:
        h_str = f'{height_mm:7.1f}' if height_mm is not None else '   --  '
        global_line = f'global mm (mic origin): x={global_xy[0]:7.1f}  y={global_xy[1]:7.1f}  z={h_str}'
    else:
        global_line = 'global mm (mic origin): --'

    if frame_xy is not None:
        frame_line = f'frame mm (corner origin): x={frame_xy[0]:7.1f}  y={frame_xy[1]:7.1f}'
    else:
        frame_line = 'frame mm (corner origin): --'

    lines = [
        f'detected: {"yes" if detected else "no"}',
        f'2D px : x={x_px:5d}  y={y_px:5d}' if detected else '2D px : --',
        f'3D mm : x={pos_x:7.1f}  y={pos_y:7.1f}  z={pos_z:7.1f}' if detected else '3D mm : --',
        local_line,
        global_line,
        frame_line,
        calib_line,
    ]
    box_h = 22 * len(lines) + 16
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (480, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 26 + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description='Index fingertip 2D/3D trajectory viewer')
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--output', type=Path, default=None,
                         help='CSV filename (default: index_trajectory_<timestamp>.csv). '
                              'Relative paths are placed under trajectory_sessions/; '
                              'absolute paths are used as-is.')
    parser.add_argument('--trail-length', type=int, default=60,
                         help='number of past points kept in the on-screen fading trail (default 60)')
    parser.add_argument('--no-skeleton', action='store_true',
                         help='skip the full hand-skeleton overlay (trail + tip marker only)')
    args = parser.parse_args()

    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)
    out_name = args.output or Path(f'index_trajectory_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    out_path = out_name if out_name.is_absolute() else TRAJECTORY_DIR / out_name

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(f'[ERROR] could not open camera index={args.camera_index}')
        return

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    autofocus_state = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    print(f'[CAM] autofocus off requested — camera reports autofocus={autofocus_state} '
          f'(0 = confirmed off; some cameras ignore this and keep autofocusing anyway)')

    tracker = MultiFingertipIMUTracker(max_num_hands=1)
    trail = deque(maxlen=max(2, args.trail_length))

    # Reuse a previous calibration if one exists (shared with
    # unified_collector_final.py) — press 'c' any time to recalibrate.
    calibration = load_calibration()
    if calibration is not None:
        print(f'[CALIBRATE] loaded existing calibration from {calibration.get("timestamp", "?")} '
              f'(scale_factor={calibration.get("scale_factor"):.4f}) — press c to redo')

    csv_file = open(out_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(TRAJECTORY_CSV_HEADER)

    cv2.namedWindow(WINDOW_NAME)
    print(f'[RUN] logging index-finger trajectory to {out_path} — press q or ESC to stop, '
          f'c to calibrate (local scale + mic global origin)')
    t0 = time.perf_counter()
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            h, w = frame.shape[:2]
            ts = time.perf_counter() - t0

            records = tracker.update(frame, timestamp=ts)
            traj = compute_trajectory(tracker, records, w, h, calibration)
            index_record = traj['index_record']

            if index_record.detected and traj['x_px'] is not None:
                trail.append((traj['x_px'], traj['y_px']))

            writer.writerow(trajectory_csv_row(ts, traj))
            frame_count += 1
            if frame_count % FLUSH_EVERY_N_FRAMES == 0:
                csv_file.flush()

            if not args.no_skeleton:
                tracker.draw(frame)

            # Fading trail: older points drawn thinner and dimmer.
            n = len(trail)
            for i in range(1, n):
                alpha = i / n
                thickness = max(1, int(alpha * 4))
                color = tuple(int(c * alpha) for c in TRAIL_COLOR)
                cv2.line(frame, trail[i - 1], trail[i], color, thickness)
            if trail:
                cv2.circle(frame, trail[-1], 6, TRAIL_COLOR, -1)

            _draw_readout(frame, index_record.detected, traj['x_px'], traj['y_px'],
                          index_record.pos_x, index_record.pos_y, index_record.pos_z,
                          traj['local_vec'], traj['global_xy'], traj['frame_xy'],
                          traj['height_mm'], calibration)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('c'):
                new_calibration = run_calibration(tracker, frame, WINDOW_NAME)
                if new_calibration is not None:
                    calibration = new_calibration
    except KeyboardInterrupt:
        print('\n[RUN] interrupted by Ctrl+C')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        csv_file.close()
        print(f'[DONE] saved trajectory -> {out_path}')


if __name__ == '__main__':
    main()
