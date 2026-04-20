import time
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class FingerState:
    landmarks: list
    index_tip_px: Tuple[int, int] = (0, 0)
    middle_tip_px: Tuple[int, int] = (0, 0)
    thumb_tip_px: Tuple[int, int] = (0, 0)
    screen_x: int = 0
    screen_y: int = 0
    middle_screen_x: int = 0
    middle_screen_y: int = 0
    pinch_dist: float = 999.0
    middle_pinch_dist: float = 999.0
    two_finger_dist: float = 999.0
    index_up: bool = False
    middle_up: bool = False
    ring_up: bool = False
    pinky_up: bool = False
    thumb_up: bool = False
    ts: float = field(default_factory=time.time)


@dataclass
class ActionEvent:
    name: str
    screen_pos: Tuple[int, int] = (0, 0)
    delta: Tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0
    extra: dict = field(default_factory=dict)
