from collections import deque
from typing import Tuple

import numpy as np


class CoordinateMapper:
    """Camera pixel -> screen pixel with ROI trimming and smoothing."""

    def __init__(self, cam_w, cam_h, screen_w, screen_h, roi_margin=0.15):
        self.screen_w = screen_w
        self.screen_h = screen_h
        mx = int(cam_w * roi_margin)
        my = int(cam_h * roi_margin)
        self.roi_x0 = mx
        self.roi_x1 = cam_w - mx
        self.roi_y0 = my
        self.roi_y1 = cam_h - my
        self.roi_w = self.roi_x1 - self.roi_x0
        self.roi_h = self.roi_y1 - self.roi_y0
        # Slightly longer window + median reduces occasional landmark spikes.
        self._sx = deque(maxlen=7)
        self._sy = deque(maxlen=7)

    def map(self, px: int, py: int, smooth: bool = True) -> Tuple[int, int]:
        cx = max(self.roi_x0, min(self.roi_x1, px))
        cy = max(self.roi_y0, min(self.roi_y1, py))
        nx = 1.0 - (cx - self.roi_x0) / self.roi_w
        ny = (cy - self.roi_y0) / self.roi_h
        sx = int(nx * self.screen_w)
        sy = int(ny * self.screen_h)
        if smooth:
            self._sx.append(sx)
            self._sy.append(sy)
            sx = int(np.median(self._sx))
            sy = int(np.median(self._sy))
        return sx, sy

    def roi_rect(self):
        return self.roi_x0, self.roi_y0, self.roi_x1, self.roi_y1
