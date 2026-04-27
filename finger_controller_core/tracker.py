import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from .mapping import CoordinateMapper
from .model import MODEL_PATH
from .types import FingerState
from .utils import normalize_rotation_deg

_LM = {
    "THUMB_TIP": 4,
    "THUMB_IP": 3,
    "INDEX_MCP": 5,
    "INDEX_TIP": 8,
    "MIDDLE_MCP": 9,
    "MIDDLE_TIP": 12,
    "RING_MCP": 13,
    "RING_TIP": 16,
    "PINKY_MCP": 17,
    "PINKY_TIP": 20,
}

_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


class HandTracker:
    def __init__(
        self,
        cam_w: int,
        cam_h: int,
        mapper: CoordinateMapper,
        camera_rotation_deg: int = 0,
        mirror_input: bool = True,
    ):
        self.cam_w = cam_w
        self.cam_h = cam_h
        self.mapper = mapper
        self.camera_rotation_deg = normalize_rotation_deg(camera_rotation_deg)
        self.mirror_input = mirror_input

        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._ts_us = 0

    def transform_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        frame = bgr_frame
        if self.camera_rotation_deg == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.camera_rotation_deg == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.camera_rotation_deg == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.mirror_input:
            frame = cv2.flip(frame, 1)

        return frame

    def process(self, bgr_frame: np.ndarray) -> Tuple[list[FingerState], np.ndarray]:
        frame = self.transform_frame(bgr_frame)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._ts_us = max(self._ts_us + 1, int(time.time() * 1_000_000))
        result = self.landmarker.detect_for_video(mp_image, self._ts_us)

        if not result.hand_landmarks:
            return [], frame

        states = []
        for lm_raw in result.hand_landmarks:
            lm = [(p.x, p.y, p.z) for p in lm_raw]

            def px(idx):
                return int(lm[idx][0] * w), int(lm[idx][1] * h)

            def dist(a, b):
                return float(np.hypot(a[0] - b[0], a[1] - b[1]))

            wrist = lm[0]

            def is_up(tip_i, mcp_i):
                # Orientation-agnostic: a finger is considered extended when
                # the tip is farther from wrist than the MCP joint.
                tip = lm[tip_i]
                mcp = lm[mcp_i]
                tip_dist = float(np.hypot(tip[0] - wrist[0], tip[1] - wrist[1]))
                mcp_dist = float(np.hypot(mcp[0] - wrist[0], mcp[1] - wrist[1]))
                return tip_dist > (mcp_dist + 0.02)

            index_tip = px(_LM["INDEX_TIP"])
            middle_tip = px(_LM["MIDDLE_TIP"])
            thumb_tip = px(_LM["THUMB_TIP"])

            sx, sy = self.mapper.map(*index_tip)
            msx, msy = self.mapper.map(*middle_tip)

            fs = FingerState(
                landmarks=lm,
                index_tip_px=index_tip,
                middle_tip_px=middle_tip,
                thumb_tip_px=thumb_tip,
                screen_x=sx,
                screen_y=sy,
                middle_screen_x=msx,
                middle_screen_y=msy,
                pinch_dist=dist(index_tip, thumb_tip),
                middle_pinch_dist=dist(middle_tip, thumb_tip),
                two_finger_dist=dist(index_tip, middle_tip),
                index_up=is_up(_LM["INDEX_TIP"], _LM["INDEX_MCP"]),
                middle_up=is_up(_LM["MIDDLE_TIP"], _LM["MIDDLE_MCP"]),
                ring_up=is_up(_LM["RING_TIP"], _LM["RING_MCP"]),
                pinky_up=is_up(_LM["PINKY_TIP"], _LM["PINKY_MCP"]),
                thumb_up=is_up(_LM["THUMB_TIP"], _LM["THUMB_IP"]),
            )
            states.append(fs)

            pts = [(int(p[0] * w), int(p[1] * h)) for p in lm]
            for a, b in _CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 220, 120), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (255, 255, 255), -1)

            cv2.circle(frame, index_tip, 10, (0, 200, 255), -1)

        rx0, ry0, rx1, ry1 = self.mapper.roi_rect()
        cv2.rectangle(frame, (rx0, ry0), (rx1, ry1), (0, 255, 100), 1)
        if states:
            cv2.putText(
                frame,
                f"Hands: {len(states)} Screen {states[0].screen_x},{states[0].screen_y}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 100),
                2,
            )

        return states, frame

    def close(self):
        self.landmarker.close()
