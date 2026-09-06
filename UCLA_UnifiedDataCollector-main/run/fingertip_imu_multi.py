"""
fingertip_imu_multi.py
────────────────────────────────────────────────────────────────────────────
MediaPipe-based module for estimating fingertip IMU (acceleration + gyro)
for all five fingers.

This extends fingertip_imu_tracker.py (single index finger) to all five
fingers. The underlying computation (constructing the local coordinate
frame, computing acceleration/angular velocity, noise mitigation) is
unchanged; only the structure is extended so that each finger maintains its
own independent state (position history, rotation matrix, EMA).

Three-point mapping (proximal joint, distal joint, tip) per finger:
  thumb  : MCP(2)  - IP(3)  - TIP(4)   *thumb has no PIP, so MCP-IP-TIP is used
  index  : PIP(6)  - DIP(7) - TIP(8)
  middle : PIP(10) - DIP(11)- TIP(12)
  ring   : PIP(14) - DIP(15)- TIP(16)
  pinky  : PIP(18) - DIP(19)- TIP(20)

Usage when integrating with unified_collector.py:
  tracker = MultiFingertipIMUTracker(smoothing_window=3, gravity_mm_s2=..., ema_alpha=0.2)
  records = tracker.update(frame_bgr, timestamp=_offset())  # list[FingertipIMURecord], 5 entries
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FingertipIMURecord:
    timestamp: float          # passed through as given at call time (unified_collector uses _offset())
    finger: str                # "thumb" / "index" / "middle" / "ring" / "pinky"
    hand_label: str            # "Left" / "Right"
    detected: bool
    accel_x: float              # mm/s^2, in the fingertip's local coordinate frame
    accel_y: float
    accel_z: float
    gyro_x: float                # rad/s, in the fingertip's local coordinate frame
    gyro_y: float
    gyro_z: float
    pos_x: float                  # mm, world-frame position (for reference/debugging)
    pos_y: float
    pos_z: float


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n >= 1e-8 else fallback


def _fingertip_orientation(prev2: np.ndarray, prev1: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """Builds a local segment coordinate frame (rotation matrix) from the three points
    prev2->prev1->tip. Same principle as fingertip_imu_tracker.py."""
    z_axis = _normalize(tip - prev1, fallback=np.array([0.0, 0.0, 1.0]))
    ref_dir = _normalize(prev1 - prev2, fallback=z_axis)

    x_raw = np.cross(ref_dir, z_axis)
    if np.linalg.norm(x_raw) < 1e-6:
        x_raw = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
        if np.linalg.norm(x_raw) < 1e-6:
            x_raw = np.cross(np.array([1.0, 0.0, 0.0]), z_axis)
    x_axis = _normalize(x_raw, fallback=np.array([1.0, 0.0, 0.0]))
    y_axis = np.cross(z_axis, x_axis)

    return np.column_stack([x_axis, y_axis, z_axis])


def _angular_velocity_body(R_prev: np.ndarray, R_curr: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0:
        return np.zeros(3)
    R_rel = R_prev.T @ R_curr
    cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    axis = np.array([
        R_rel[2, 1] - R_rel[1, 2],
        R_rel[0, 2] - R_rel[2, 0],
        R_rel[1, 0] - R_rel[0, 1],
    ]) / (2.0 * np.sin(theta))
    return axis * theta / dt


def gravity_vector_from_camera_tilt(
    pitch_deg: float, roll_deg: float = 0.0, magnitude_mm_s2: float = 9810.0,
) -> np.ndarray:
    """Computes the gravity vector from a fixed camera tilt (same as
    fingertip_imu_tracker.py; still needs empirical validation)."""
    pitch, roll = np.radians(pitch_deg), np.radians(roll_deg)
    down_level = np.array([0.0, 1.0, 0.0])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    return (Rz @ Rx @ down_level) * magnitude_mm_s2


class MultiFingertipIMUTracker:
    """Independently runs the same computation as FingertipIMUTracker for each of the five fingers."""

    FINGER_JOINTS = {
        "thumb":  (2, 3, 4),
        "index":  (6, 7, 8),
        "middle": (10, 11, 12),
        "ring":   (14, 15, 16),
        "pinky":  (18, 19, 20),
    }
    FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

    def __init__(
        self,
        max_num_hands: int = 1,
        detection_confidence: float = 0.6,
        tracking_confidence: float = 0.6,
        smoothing_window: int = 3,
        gravity_mm_s2: Optional[np.ndarray] = None,
        ema_alpha: float = 0.2,
    ):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            model_complexity=1,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.smoothing_window = max(1, smoothing_window)
        self.gravity = np.asarray(gravity_mm_s2, dtype=np.float64) if gravity_mm_s2 is not None else None
        self.ema_alpha = min(max(ema_alpha, 1e-3), 1.0)

        self._state = {name: self._new_state() for name in self.FINGER_ORDER}
        self.last_results = None

    def _new_state(self) -> dict:
        return {
            "pos_history": deque(maxlen=max(3, self.smoothing_window + 2)),
            "landmark_hist": deque(maxlen=max(2, self.smoothing_window)),
            "last_velocity": None,
            "last_R": None,
            "last_timestamp": None,
            "accel_ema": None,
            "gyro_ema": None,
        }

    def close(self):
        self._hands.close()

    def _smoothed_diff(self, values: np.ndarray, times: np.ndarray) -> np.ndarray:
        if len(values) < 2:
            return np.zeros(3)
        diffs = []
        for i in range(1, len(values)):
            dt = times[i] - times[i - 1]
            if dt > 0:
                diffs.append((values[i] - values[i - 1]) / dt)
        if not diffs:
            return np.zeros(3)
        return np.mean(diffs[-self.smoothing_window:], axis=0)

    def update(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> list:
        if timestamp is None:
            timestamp = time.time()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        self.last_results = results

        if not results.multi_hand_world_landmarks:
            return [
                FingertipIMURecord(
                    timestamp=timestamp, finger=name, hand_label="", detected=False,
                    accel_x=float("nan"), accel_y=float("nan"), accel_z=float("nan"),
                    gyro_x=float("nan"), gyro_y=float("nan"), gyro_z=float("nan"),
                    pos_x=float("nan"), pos_y=float("nan"), pos_z=float("nan"),
                )
                for name in self.FINGER_ORDER
            ]

        world_landmarks = results.multi_hand_world_landmarks[0]
        label = results.multi_handedness[0].classification[0].label

        def lm_mm(idx: int) -> np.ndarray:
            lm = world_landmarks.landmark[idx]
            return np.array([lm.x, lm.y, lm.z], dtype=np.float64) * 1000.0

        records = []
        for name in self.FINGER_ORDER:
            j0, j1, j2 = self.FINGER_JOINTS[name]
            state = self._state[name]

            prev2, prev1, tip = lm_mm(j0), lm_mm(j1), lm_mm(j2)

            # Smooth the raw landmarks themselves (mitigates gyro noise)
            state["landmark_hist"].append(np.stack([prev2, prev1, tip]))
            smoothed = np.mean(state["landmark_hist"], axis=0)
            prev2, prev1, tip = smoothed[0], smoothed[1], smoothed[2]

            # Position/velocity/acceleration (world frame, mm)
            state["pos_history"].append((timestamp, tip))
            times = np.array([t for t, _ in state["pos_history"]])
            positions = np.array([p for _, p in state["pos_history"]])
            velocity_world = self._smoothed_diff(positions, times)

            if state["last_velocity"] is not None and len(state["pos_history"]) >= 2:
                dt = times[-1] - times[-2]
                accel_world = (velocity_world - state["last_velocity"]) / dt if dt > 0 else np.zeros(3)
            else:
                accel_world = np.zeros(3)
            state["last_velocity"] = velocity_world

            # Local coordinate frame + acceleration transform
            R_curr = _fingertip_orientation(prev2, prev1, tip)
            specific_force_world = accel_world.copy()
            if self.gravity is not None:
                specific_force_world = specific_force_world - self.gravity
            accel_local = R_curr.T @ specific_force_world

            # Angular velocity
            if state["last_R"] is not None and state["last_timestamp"] is not None:
                dt = timestamp - state["last_timestamp"]
                gyro_local = _angular_velocity_body(state["last_R"], R_curr, dt)
            else:
                gyro_local = np.zeros(3)
            state["last_R"] = R_curr
            state["last_timestamp"] = timestamp

            # EMA smoothing
            state["accel_ema"] = accel_local if state["accel_ema"] is None else (
                self.ema_alpha * accel_local + (1 - self.ema_alpha) * state["accel_ema"])
            state["gyro_ema"] = gyro_local if state["gyro_ema"] is None else (
                self.ema_alpha * gyro_local + (1 - self.ema_alpha) * state["gyro_ema"])

            records.append(FingertipIMURecord(
                timestamp=timestamp, finger=name, hand_label=label, detected=True,
                accel_x=state["accel_ema"][0], accel_y=state["accel_ema"][1], accel_z=state["accel_ema"][2],
                gyro_x=state["gyro_ema"][0], gyro_y=state["gyro_ema"][1], gyro_z=state["gyro_ema"][2],
                pos_x=tip[0], pos_y=tip[1], pos_z=tip[2],
            ))

        return records

    def draw(self, frame_bgr: np.ndarray) -> np.ndarray:
        """For debug visualization (hand skeleton). Can be omitted in integration."""
        if self.last_results and self.last_results.multi_hand_landmarks:
            for hand_landmarks in self.last_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame_bgr, hand_landmarks, self._mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
        return frame_bgr

    def draw_axes(self, frame_bgr: np.ndarray, axis_length_px: int = 40) -> np.ndarray:
        """Draws the computed local coordinate axes (x=green, y=purple, z=red) at each
        fingertip. Uses an orthographic-projection approximation."""
        if not (self.last_results and self.last_results.multi_hand_landmarks):
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        colors = {"x": (80, 200, 80), "y": (200, 80, 200), "z": (50, 50, 230)}
        landmarks_2d = self.last_results.multi_hand_landmarks[0].landmark

        for name in self.FINGER_ORDER:
            state = self._state[name]
            if state["last_R"] is None:
                continue
            tip_idx = self.FINGER_JOINTS[name][2]
            lm = landmarks_2d[tip_idx]
            origin = (int(lm.x * w), int(lm.y * h))
            for i, axis_name in enumerate(("x", "y", "z")):
                vec = state["last_R"][:, i]
                end = (int(origin[0] + vec[0] * axis_length_px), int(origin[1] + vec[1] * axis_length_px))
                cv2.arrowedLine(frame_bgr, origin, end, colors[axis_name], 1, tipLength=0.35)
        return frame_bgr