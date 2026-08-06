"""
trajectory_calibration.py
────────────────────────────────────────────────────────────────────────────
Shared index-fingertip trajectory computation + mic-anchored calibration,
used by both index_trajectory_viewer.py (standalone tool) and
unified_collector_final.py (main data-collection pipeline), so both compute
and log trajectory data identically instead of maintaining two copies.

Calibration (two parts, saved together to run/calibration.json):
  (A) Local wrist-relative scale — the real physically-measured wrist-to-
      middle-knuckle distance vs. what MediaPipe reports gives a
      scale-correction factor, correcting MediaPipe's generic hand-size
      assumption to this user's actual hand. Applied to an explicitly-computed
      wrist-relative local position for the index fingertip (rather than
      trusting MediaPipe's own internal, undocumented world-landmark origin).
  (B) Global mic-anchored origin — two clicks on the contact mic's visible
      diameter (a known 30mm reference) set a fixed global origin + pixels-
      to-mm scale for the writing surface. Unlike (A), this doesn't reset
      every frame, so it can actually capture whole-hand translation.
      Optionally also a camera-to-surface distance measurement, which —
      combined with (A)'s known real wrist-to-knuckle distance and its
      apparent pixel size — derives an approximate focal length, enabling a
      rough height/Z estimate above the surface via depth-from-known-size.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

INDEX_TIP_LANDMARK = 8    # MediaPipe Hands landmark index for the index fingertip
WRIST_LANDMARK = 0        # MediaPipe Hands landmark index for the wrist
MIDDLE_MCP_LANDMARK = 9   # MediaPipe Hands landmark index for the middle-finger base knuckle
MIC_DIAMETER_MM = 30.0    # https://www.amazon.com/dp/B0BZJLGKCQ contact mic disc diameter
CALIBRATION_FILE = Path(__file__).resolve().parent / 'calibration.json'

TRAJECTORY_CSV_HEADER = [
    'timestamp_sec', 'detected',
    'x_px', 'y_px', 'x_norm', 'y_norm',
    'pos_x_mm', 'pos_y_mm', 'pos_z_mm',
    'local_x_mm', 'local_y_mm', 'local_z_mm', 'calibrated',
    'global_x_mm', 'global_y_mm', 'height_mm',
    'frame_x_mm', 'frame_y_mm',
]


def world_landmark_mm(tracker, idx: int):
    """Raw (x, y, z) in mm for one landmark, read directly from MediaPipe's
    world landmarks — bypasses the tracker's own smoothed/EMA state so the
    wrist-relative vector is an explicit, transparent computation rather than
    depending on MediaPipe's internal, undocumented origin."""
    results = tracker.last_results
    if not results or not results.multi_hand_world_landmarks:
        return None
    lm = results.multi_hand_world_landmarks[0].landmark[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64) * 1000.0


def landmark_px(tracker, idx: int, w: int, h: int):
    """Raw 2D pixel position for one landmark, from MediaPipe's normalized
    image-plane landmarks (NOT world landmarks — those normalize out the
    perspective/scale information this needs to detect apparent-size changes
    for depth-from-known-size)."""
    results = tracker.last_results
    if not results or not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0].landmark[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float64)


def click_two_points(window_name: str, frame, instructions: str):
    """Shows `frame` in `window_name` and waits for exactly two left-clicks
    (e.g. marking opposite edges of the contact mic's visible diameter).
    Esc cancels. Returns [(x1,y1), (x2,y2)] or None."""
    points = []

    def _on_click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.setMouseCallback(window_name, _on_click)
    cancelled = False
    while len(points) < 2:
        display = frame.copy()
        cv2.putText(display, instructions, (10, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        for p in points:
            cv2.circle(display, p, 5, (0, 255, 255), -1)
        cv2.imshow(window_name, display)
        if cv2.waitKey(20) & 0xFF == 27:
            cancelled = True
            break

    if not cancelled:
        # Brief final display with the measured diameter drawn, for visual confirmation.
        display = frame.copy()
        cv2.line(display, points[0], points[1], (0, 255, 255), 1)
        for p in points:
            cv2.circle(display, p, 5, (0, 255, 255), -1)
        cv2.imshow(window_name, display)
        cv2.waitKey(400)

    cv2.setMouseCallback(window_name, lambda *a: None)
    if cancelled:
        print('[CALIBRATE] cancelled')
        return None
    return points


def prompt_number(window_name: str, frame, prompt: str) -> str | None:
    """Shows `frame` in `window_name` with an on-screen numeric entry —
    digits and '.' type, Backspace deletes, Enter confirms, Esc cancels.
    Deliberately not input()/stdin: this needs to work identically whether
    run_calibration() is called in-process (index_trajectory_viewer.py) or
    inside a multiprocessing subprocess (unified_collector_final.py's
    pre-recording calibration) — the latter was found to raise EOFError on
    input(), since a 'spawn'-started child doesn't reliably inherit a usable
    stdin. Returns the typed string (possibly '' if confirmed blank), or
    None if cancelled."""
    buf = ''
    while True:
        display = frame.copy()
        h = display.shape[0]
        cv2.putText(display, prompt, (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f'> {buf}_', (10, h - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, 'digits/. to type, Backspace to delete, Enter to confirm, Esc to cancel',
                    (10, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            return None
        if key in (13, 10):
            return buf
        if key in (8, 127):
            buf = buf[:-1]
        elif key == ord('.') and '.' not in buf:
            buf += '.'
        elif ord('0') <= key <= ord('9'):
            buf += chr(key)
        # any other key is ignored


def load_calibration() -> dict | None:
    """Loads run/calibration.json if it exists and is valid. scale_factor is
    a personal constant (how far off MediaPipe's generic hand-size assumption
    is from your actual hand), not something that needs re-measuring every
    run, so callers should reuse this instead of forcing recalibration."""
    if not CALIBRATION_FILE.exists():
        return None
    try:
        with open(CALIBRATION_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, TypeError) as e:
        print(f'[CALIBRATE] could not load {CALIBRATION_FILE} ({e}) — starting uncalibrated')
        return None


def run_calibration(tracker, frame, window_name: str) -> dict | None:
    """Does the mic-diameter click step first, then both physical-measurement
    prompts — all on-screen in the same window (see prompt_number), so
    there's no alternating between the video window and a terminal, and no
    dependency on stdin/input() (which isn't reliably available when this
    runs inside a multiprocessing subprocess).

    (A) local wrist-relative scale, then (B) global mic-anchored origin —
    see module docstring. Saves the combined result to CALIBRATION_FILE.
    """
    h, w = frame.shape[:2]

    # ---- gather everything from the current frame first, no prompts yet ----
    wrist_mm = world_landmark_mm(tracker, WRIST_LANDMARK)
    mcp_mm = world_landmark_mm(tracker, MIDDLE_MCP_LANDMARK)
    if wrist_mm is None or mcp_mm is None:
        print('[CALIBRATE] no hand detected right now — hold your hand steady in frame and try again')
        return None
    reported_mm = float(np.linalg.norm(mcp_mm - wrist_mm))

    wrist_px = landmark_px(tracker, WRIST_LANDMARK, w, h)
    mcp_px = landmark_px(tracker, MIDDLE_MCP_LANDMARK, w, h)
    apparent_px_at_calibration = (float(np.linalg.norm(mcp_px - wrist_px))
                                  if wrist_px is not None and mcp_px is not None else None)

    # ---- the one on-screen step: click the mic's diameter ----
    mic_points = click_two_points(window_name, frame,
                                  'click 2 opposite edges of the mic diameter, Esc to skip')
    mic_center_px = mic_diameter_px = mm_per_pixel = None
    if mic_points is not None:
        p0 = np.array(mic_points[0], dtype=np.float64)
        p1 = np.array(mic_points[1], dtype=np.float64)
        mic_diameter_px = float(np.linalg.norm(p1 - p0))
        if mic_diameter_px <= 0:
            print('[CALIBRATE] invalid click distance — skipping global-origin calibration')
            mic_diameter_px = None
        else:
            mic_center_px = ((p0 + p1) / 2).tolist()
            mm_per_pixel = MIC_DIAMETER_MM / mic_diameter_px

    # ---- now both physical-measurement prompts, on-screen, back to back ----
    print(f'[CALIBRATE] MediaPipe reports wrist-to-middle-knuckle distance = {reported_mm:.2f}mm')
    measured_str = prompt_number(
        window_name, frame,
        f'MediaPipe reports wrist-to-knuckle = {reported_mm:.2f}mm. Enter the physically measured distance (mm):')
    if measured_str is None:
        print('[CALIBRATE] cancelled')
        return None
    try:
        measured_mm = float(measured_str)
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

    focal_length_px = None
    if mm_per_pixel is not None:
        calibration.update({
            'mic_center_px': mic_center_px,
            'mic_diameter_px': mic_diameter_px,
            'mic_diameter_mm': MIC_DIAMETER_MM,
            'mm_per_pixel': mm_per_pixel,
        })
        if apparent_px_at_calibration and apparent_px_at_calibration > 0:
            raw = prompt_number(
                window_name, frame,
                'Measure camera-to-surface distance now (mm), hand resting on surface. '
                'Enter blank + Enter (or Esc) to skip height/Z tracking:')
            camera_to_surface_mm = None
            if raw:
                try:
                    camera_to_surface_mm = float(raw)
                except ValueError:
                    camera_to_surface_mm = None
            if camera_to_surface_mm and camera_to_surface_mm > 0:
                focal_length_px = apparent_px_at_calibration * camera_to_surface_mm / measured_mm
                calibration.update({
                    'camera_to_surface_mm': camera_to_surface_mm,
                    'focal_length_px': focal_length_px,
                    'wrist_mcp_apparent_px_at_calibration': apparent_px_at_calibration,
                })
        print(f'[CALIBRATE] mic origin set at {mic_center_px} px, '
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


def compute_trajectory(tracker, records, w: int, h: int, calibration: dict | None) -> dict:
    """Computes every trajectory quantity for the index finger from this
    frame's already-computed `records`/tracker state. Returns a dict with:
      index_record, x_px, y_px, x_norm, y_norm  (raw 2D)
      local_vec       (wrist-relative 3D mm, scale-corrected if calibrated)
      global_xy       (mic-anchored 2D mm, None until calibrated)
      frame_xy        (frame-corner-anchored 2D mm, None until calibrated)
      height_mm       (depth-from-known-size height estimate, None unless
                       the camera-to-surface distance was also given)
      calibrated      (bool)
    """
    index_record = next(r for r in records if r.finger == 'index')

    x_px = y_px = x_norm = y_norm = None
    if index_record.detected and tracker.last_results.multi_hand_landmarks:
        lm = tracker.last_results.multi_hand_landmarks[0].landmark[INDEX_TIP_LANDMARK]
        x_norm, y_norm = lm.x, lm.y
        x_px, y_px = int(x_norm * w), int(y_norm * h)

    # Explicit wrist-relative local vector (Part A) — computed directly from
    # raw world landmarks rather than trusting MediaPipe's own internal,
    # undocumented world-landmark origin. Scale-corrected once calibrated.
    local_vec = None
    wrist_mm = world_landmark_mm(tracker, WRIST_LANDMARK)
    index_tip_mm = world_landmark_mm(tracker, INDEX_TIP_LANDMARK)
    if wrist_mm is not None and index_tip_mm is not None:
        local_vec = index_tip_mm - wrist_mm
        if calibration is not None:
            local_vec = local_vec * calibration['scale_factor']

    # Global mic-anchored (x, y) via the calibrated pixel origin/scale
    # (Part B) — fixed for the whole session, unlike local_vec above, so this
    # one can actually reflect whole-hand translation.
    global_xy = None
    frame_xy = None
    height_mm = None
    if calibration is not None and calibration.get('mm_per_pixel') is not None and x_px is not None:
        mm_per_pixel = calibration['mm_per_pixel']
        origin_x_px, origin_y_px = calibration['mic_center_px']
        global_xy = ((x_px - origin_x_px) * mm_per_pixel,
                     (y_px - origin_y_px) * mm_per_pixel)

        # Same scale, but referenced to the frame's top-left corner (pixel
        # 0,0) instead of the mic — a camera-referenced translation rather
        # than a physical-object-referenced one.
        frame_xy = (x_px * mm_per_pixel, y_px * mm_per_pixel)

        # Height above the calibration surface, via depth-from-known-size:
        # compare the wrist-to-knuckle apparent pixel size now against its
        # size at calibration to estimate how distance-from-camera changed.
        if calibration.get('focal_length_px') is not None:
            wrist_px = landmark_px(tracker, WRIST_LANDMARK, w, h)
            mcp_px = landmark_px(tracker, MIDDLE_MCP_LANDMARK, w, h)
            if wrist_px is not None and mcp_px is not None:
                apparent_px_now = float(np.linalg.norm(mcp_px - wrist_px))
                if apparent_px_now > 0:
                    distance_now_mm = (calibration['focal_length_px']
                                       * calibration['measured_mm'] / apparent_px_now)
                    height_mm = calibration['camera_to_surface_mm'] - distance_now_mm

    return {
        'index_record': index_record,
        'x_px': x_px, 'y_px': y_px, 'x_norm': x_norm, 'y_norm': y_norm,
        'local_vec': local_vec, 'global_xy': global_xy, 'frame_xy': frame_xy,
        'height_mm': height_mm, 'calibrated': calibration is not None,
    }


def trajectory_csv_row(ts: float, traj: dict) -> list:
    """Formats a compute_trajectory() result into a CSV row matching
    TRAJECTORY_CSV_HEADER."""
    index_record = traj['index_record']
    local_vec = traj['local_vec']
    global_xy = traj['global_xy']
    frame_xy = traj['frame_xy']
    height_mm = traj['height_mm']
    return [
        f'{ts:.6f}', int(index_record.detected),
        traj['x_px'] if traj['x_px'] is not None else '',
        traj['y_px'] if traj['y_px'] is not None else '',
        f'{traj["x_norm"]:.6f}' if traj['x_norm'] is not None else '',
        f'{traj["y_norm"]:.6f}' if traj['y_norm'] is not None else '',
        f'{index_record.pos_x:.3f}' if index_record.detected else '',
        f'{index_record.pos_y:.3f}' if index_record.detected else '',
        f'{index_record.pos_z:.3f}' if index_record.detected else '',
        f'{local_vec[0]:.3f}' if local_vec is not None else '',
        f'{local_vec[1]:.3f}' if local_vec is not None else '',
        f'{local_vec[2]:.3f}' if local_vec is not None else '',
        int(traj['calibrated']),
        f'{global_xy[0]:.3f}' if global_xy is not None else '',
        f'{global_xy[1]:.3f}' if global_xy is not None else '',
        f'{height_mm:.3f}' if height_mm is not None else '',
        f'{frame_xy[0]:.3f}' if frame_xy is not None else '',
        f'{frame_xy[1]:.3f}' if frame_xy is not None else '',
    ]
