"""
index_trajectory_viewer.py
────────────────────────────────────────────────────────────────────────────
Standalone demo: tracks the index fingertip's 2D (pixel/image-plane) and 3D
(mm, world-frame) trajectory using MediaPipe Hands, via the same tracker
unified_collector_final.py uses (fingertip_imu_multi.MultiFingertipIMUTracker).

Draws a live fading trail on the camera preview and logs both positions to a
CSV file, one row per camera frame.

Trajectory CSVs are saved under trajectory_sessions/ (created alongside this
script if it doesn't exist yet), not directly in run/.

Calibration (press 'c') does two things in one pass, saved together to
trajectory_sessions/calibration.json:
  (A) Local wrist-relative scale: measures the wrist-to-middle-knuckle distance
      MediaPipe reports, asks for the same distance as physically measured on
      your own hand (ruler/calipers), and derives a scale-correction factor
      from the two — corrects for MediaPipe's generic hand-size assumption not
      matching your actual hand. Applied to an explicitly-computed
      wrist-relative local position for the index fingertip (rather than
      trusting MediaPipe's own internal, undocumented world-landmark origin).
  (B) Global mic-anchored origin: click the two opposite edges of the contact
      mic's visible diameter (a known 27mm reference) to set it as the fixed
      global origin and derive a pixels-to-mm scale for the writing surface —
      unlike (A), this doesn't reset every frame, so it can actually capture
      whole-hand translation. Optionally also measure the camera-to-surface
      distance once, which — combined with (A)'s known real wrist-to-knuckle
      distance and its apparent pixel size — derives an approximate focal
      length, enabling a rough height/Z estimate above the surface via
      depth-from-known-size.

Usage:
    python index_trajectory_viewer.py
    python index_trajectory_viewer.py --camera-index 1 --output my_traj.csv
    python index_trajectory_viewer.py --trail-length 90 --no-skeleton
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from fingertip_imu_multi import MultiFingertipIMUTracker

INDEX_TIP_LANDMARK = 8    # MediaPipe Hands landmark index for the index fingertip
WRIST_LANDMARK = 0        # MediaPipe Hands landmark index for the wrist
MIDDLE_MCP_LANDMARK = 9   # MediaPipe Hands landmark index for the middle-finger base knuckle
TRAIL_COLOR = (60, 220, 255)   # BGR
FLUSH_EVERY_N_FRAMES = 30
TRAJECTORY_DIR = Path(__file__).resolve().parent / 'trajectory_sessions'
CALIBRATION_FILE = TRAJECTORY_DIR / 'calibration.json'
WINDOW_NAME = 'Index Fingertip Trajectory'
MIC_DIAMETER_MM = 30.0   # https://www.amazon.com/dp/B0BZJLGKCQ contact mic disc diameter


def _world_landmark_mm(tracker, idx: int):
    """Raw (x, y, z) in mm for one landmark, read directly from MediaPipe's
    world landmarks — bypasses the tracker's own smoothed/EMA state so the
    wrist-relative vector below is an explicit, transparent computation
    rather than depending on MediaPipe's internal, undocumented origin."""
    results = tracker.last_results
    if not results or not results.multi_hand_world_landmarks:
        return None
    lm = results.multi_hand_world_landmarks[0].landmark[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64) * 1000.0


def _landmark_px(tracker, idx: int, w: int, h: int):
    """Raw 2D pixel position for one landmark, from MediaPipe's normalized
    image-plane landmarks (NOT world landmarks — those normalize out the
    perspective/scale information this needs to detect apparent-size changes
    for depth-from-known-size)."""
    results = tracker.last_results
    if not results or not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0].landmark[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)


def _click_two_points(frame, instructions: str):
    """Shows `frame` in the trajectory window and waits for exactly two
    left-clicks (e.g. marking opposite edges of the contact mic's visible
    diameter). Esc cancels. Returns [(x1,y1), (x2,y2)] or None."""
    points = []

    def _on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.setMouseCallback(WINDOW_NAME, _on_click)
    cancelled = False
    while len(points) < 2:
        display = frame.copy()
        cv2.putText(display, instructions, (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        for p in points:
            cv2.circle(display, p, 5, (0, 255, 255), -1)
        cv2.imshow(WINDOW_NAME, display)
        if cv2.waitKey(20) & 0xFF == 27:
            cancelled = True
            break

    if not cancelled:
        # Brief final display with the measured diameter drawn, for visual confirmation.
        display = frame.copy()
        cv2.line(display, points[0], points[1], (0, 255, 255), 1)
        for p in points:
            cv2.circle(display, p, 5, (0, 255, 255), -1)
        cv2.imshow(WINDOW_NAME, display)
        cv2.waitKey(400)

    cv2.setMouseCallback(WINDOW_NAME, lambda *a: None)
    if cancelled:
        print('[CALIBRATE] cancelled')
        return None
    return points


def _run_calibration(tracker, frame) -> dict | None:
    """(A) Freezes the current frame's wrist and middle-knuckle positions, asks
    for the real physically-measured distance between those two points, and
    derives a scale-correction factor (real_mm / mediapipe_reported_mm) for
    wrist-relative local positions.

    (B) Then asks you to click the two opposite edges of the contact mic's
    visible diameter, to set it as a fixed global origin + pixels-to-mm scale
    for the writing surface. Optionally also asks for the camera-to-surface
    distance, which — combined with (A)'s known real wrist-to-knuckle distance
    and its apparent pixel size right now — derives an approximate focal
    length (depth-from-known-size), enabling a rough height/Z estimate above
    the surface for later frames.
    """
    h, w = frame.shape[:2]

    # ---- (A) local wrist-relative scale ----
    wrist_mm = _world_landmark_mm(tracker, WRIST_LANDMARK)
    mcp_mm = _world_landmark_mm(tracker, MIDDLE_MCP_LANDMARK)
    if wrist_mm is None or mcp_mm is None:
        print('[CALIBRATE] no hand detected right now — hold your hand steady in frame and try again')
        return None

    reported_mm = float(np.linalg.norm(mcp_mm - wrist_mm))
    print(f'[CALIBRATE] MediaPipe reports wrist-to-middle-knuckle distance = {reported_mm:.2f}mm')
    try:
        measured_mm = float(input('[CALIBRATE] enter the same distance as physically measured (mm): '))
    except ValueError:
        print('[CALIBRATE] not a valid number — calibration cancelled')
        return None
    if measured_mm <= 0 or reported_mm <= 0:
        print('[CALIBRATE] distance must be positive — calibration cancelled')
        return None

    scale_factor = measured_mm / reported_mm
    calibration = {
        'scale_factor': scale_factor,
        'mediapipe_reported_mm': reported_mm,
        'measured_mm': measured_mm,
        'wrist_landmark': WRIST_LANDMARK,
        'middle_mcp_landmark': MIDDLE_MCP_LANDMARK,
        'timestamp': datetime.now().isoformat(),
    }

    # ---- (B) mic-anchored global origin + XY scale ----
    print("[CALIBRATE] now click the two opposite edges of the contact mic's "
          "visible diameter (Esc to skip global-origin calibration)")
    mic_points = _click_two_points(frame, 'click 2 opposite edges of the mic diameter, Esc to skip')
    if mic_points is not None:
        p0 = np.array(mic_points[0], dtype=np.float64)
        p1 = np.array(mic_points[1], dtype=np.float64)
        diameter_px = float(np.linalg.norm(p1 - p0))
        if diameter_px <= 0:
            print('[CALIBRATE] invalid click distance — skipping global-origin calibration')
        else:
            mm_per_pixel = MIC_DIAMETER_MM / diameter_px
            calibration.update({
                'mic_center_px': ((p0 + p1) / 2).tolist(),
                'mic_diameter_px': diameter_px,
                'mic_diameter_mm': MIC_DIAMETER_MM,
                'mm_per_pixel': mm_per_pixel,
            })

            # depth-from-known-size focal length, reusing (A)'s wrist-to-knuckle
            # real distance as the size reference
            wrist_px = _landmark_px(tracker, WRIST_LANDMARK, w, h)
            mcp_px = _landmark_px(tracker, MIDDLE_MCP_LANDMARK, w, h)
            focal_length_px = None
            camera_to_surface_mm = None
            if wrist_px is not None and mcp_px is not None:
                apparent_px_at_calibration = float(np.linalg.norm(mcp_px - wrist_px))
                try:
                    raw = input('[CALIBRATE] measure camera-to-surface distance right now (mm), '
                                'or leave blank to skip height/Z tracking: ').strip()
                    camera_to_surface_mm = float(raw) if raw else None
                except ValueError:
                    camera_to_surface_mm = None
                if camera_to_surface_mm and camera_to_surface_mm > 0 and apparent_px_at_calibration > 0:
                    focal_length_px = apparent_px_at_calibration * camera_to_surface_mm / measured_mm
                    calibration.update({
                        'camera_to_surface_mm': camera_to_surface_mm,
                        'focal_length_px': focal_length_px,
                        'wrist_mcp_apparent_px_at_calibration': apparent_px_at_calibration,
                    })

            print(f'[CALIBRATE] mic origin set at {calibration["mic_center_px"]} px, '
                  f'mm_per_pixel={mm_per_pixel:.4f}'
                  + (f', focal_length_px={focal_length_px:.1f} (height/Z tracking enabled)'
                     if focal_length_px else ' (height/Z tracking skipped)'))
    else:
        print('[CALIBRATE] global-origin calibration skipped — only local (wrist-relative) values available')

    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(calibration, f, indent=2)
    print(f'[CALIBRATE] scale_factor={scale_factor:.4f} — saved to {CALIBRATION_FILE}')
    return calibration


def _draw_readout(frame, detected: bool, x_px, y_px, pos_x, pos_y, pos_z,
                   local_vec, global_xy, height_mm, calibration: dict | None):
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

    lines = [
        f'detected: {"yes" if detected else "no"}',
        f'2D px : x={x_px:5d}  y={y_px:5d}' if detected else '2D px : --',
        f'3D mm : x={pos_x:7.1f}  y={pos_y:7.1f}  z={pos_z:7.1f}' if detected else '3D mm : --',
        local_line,
        global_line,
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
    calibration: dict | None = None

    csv_file = open(out_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(['timestamp_sec', 'detected',
                      'x_px', 'y_px', 'x_norm', 'y_norm',
                      'pos_x_mm', 'pos_y_mm', 'pos_z_mm',
                      'local_x_mm', 'local_y_mm', 'local_z_mm', 'calibrated',
                      'global_x_mm', 'global_y_mm', 'height_mm'])

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
            index_record = next(r for r in records if r.finger == 'index')

            x_px = y_px = x_norm = y_norm = None
            if index_record.detected and tracker.last_results.multi_hand_landmarks:
                lm = tracker.last_results.multi_hand_landmarks[0].landmark[INDEX_TIP_LANDMARK]
                x_norm, y_norm = lm.x, lm.y
                x_px, y_px = int(x_norm * w), int(y_norm * h)
                trail.append((x_px, y_px))

            # Explicit wrist-relative local vector (Part A) — computed directly
            # from raw world landmarks rather than trusting MediaPipe's own
            # internal, undocumented world-landmark origin. Scale-corrected
            # once a calibration exists (see _run_calibration).
            local_vec = None
            wrist_mm = _world_landmark_mm(tracker, WRIST_LANDMARK)
            index_tip_mm = _world_landmark_mm(tracker, INDEX_TIP_LANDMARK)
            if wrist_mm is not None and index_tip_mm is not None:
                local_vec = index_tip_mm - wrist_mm
                if calibration is not None:
                    local_vec = local_vec * calibration['scale_factor']

            # Global mic-anchored (x, y) via the calibrated pixel origin/scale
            # (Part B) — fixed for the whole session, unlike local_vec above,
            # so this one can actually reflect whole-hand translation.
            global_xy = None
            height_mm = None
            if calibration is not None and calibration.get('mm_per_pixel') is not None and x_px is not None:
                origin_x_px, origin_y_px = calibration['mic_center_px']
                mm_per_pixel = calibration['mm_per_pixel']
                global_xy = ((x_px - origin_x_px) * mm_per_pixel,
                             (y_px - origin_y_px) * mm_per_pixel)

                # Height above the calibration surface, via depth-from-known-size:
                # compare the wrist-to-knuckle apparent pixel size now against its
                # size at calibration to estimate how distance-from-camera changed.
                if calibration.get('focal_length_px') is not None:
                    wrist_px = _landmark_px(tracker, WRIST_LANDMARK, w, h)
                    mcp_px = _landmark_px(tracker, MIDDLE_MCP_LANDMARK, w, h)
                    if wrist_px is not None and mcp_px is not None:
                        apparent_px_now = float(np.linalg.norm(mcp_px - wrist_px))
                        if apparent_px_now > 0:
                            distance_now_mm = (calibration['focal_length_px']
                                               * calibration['measured_mm'] / apparent_px_now)
                            height_mm = calibration['camera_to_surface_mm'] - distance_now_mm

            writer.writerow([
                f'{ts:.6f}', int(index_record.detected),
                x_px if x_px is not None else '',
                y_px if y_px is not None else '',
                f'{x_norm:.6f}' if x_norm is not None else '',
                f'{y_norm:.6f}' if y_norm is not None else '',
                f'{index_record.pos_x:.3f}' if index_record.detected else '',
                f'{index_record.pos_y:.3f}' if index_record.detected else '',
                f'{index_record.pos_z:.3f}' if index_record.detected else '',
                f'{local_vec[0]:.3f}' if local_vec is not None else '',
                f'{local_vec[1]:.3f}' if local_vec is not None else '',
                f'{local_vec[2]:.3f}' if local_vec is not None else '',
                int(calibration is not None),
                f'{global_xy[0]:.3f}' if global_xy is not None else '',
                f'{global_xy[1]:.3f}' if global_xy is not None else '',
                f'{height_mm:.3f}' if height_mm is not None else '',
            ])
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

            _draw_readout(frame, index_record.detected, x_px, y_px,
                          index_record.pos_x, index_record.pos_y, index_record.pos_z,
                          local_vec, global_xy, height_mm, calibration)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('c'):
                new_calibration = _run_calibration(tracker, frame)
                if new_calibration is not None:
                    calibration = new_calibration
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        csv_file.close()
        print(f'[DONE] saved trajectory -> {out_path}')


if __name__ == '__main__':
    main()
