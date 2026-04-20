import time

import cv2
import pyautogui

from .actions import ActionDispatcher
from .gestures import GestureEngine
from .mapping import CoordinateMapper
from .model import ensure_model
from .tracker import HandTracker
from .types import ActionEvent
from .utils import normalize_rotation_deg


class FingerController:
    def __init__(
        self,
        camera_index=0,
        show_preview=True,
        target_fps=25,
        process_every_n_frames=2,
        camera_rotation_deg=0,
        mirror_input=True,
    ):
        ensure_model()
        self.show_preview = show_preview
        self._frame_dt = 1.0 / target_fps
        self._process_every_n_frames = max(1, int(process_every_n_frames))
        self._frame_index = 0
        self.camera_rotation_deg = normalize_rotation_deg(camera_rotation_deg)
        self.mirror_input = mirror_input

        screen_w, screen_h = pyautogui.size()
        print(f"[FingerController] Screen: {screen_w}x{screen_h}")

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 424)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        cam_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[FingerController] Camera: {cam_w}x{cam_h}")

        if self.camera_rotation_deg in (90, 270):
            map_w, map_h = cam_h, cam_w
        else:
            map_w, map_h = cam_w, cam_h

        self.mapper = CoordinateMapper(map_w, map_h, screen_w, screen_h)
        self.tracker = HandTracker(
            cam_w,
            cam_h,
            self.mapper,
            camera_rotation_deg=self.camera_rotation_deg,
            mirror_input=self.mirror_input,
        )
        self.gesture = GestureEngine()
        self.dispatch = ActionDispatcher()
        self._running = False

    def inject_action(self, action_name: str, screen_x: int, screen_y: int, **kwargs):
        ev = ActionEvent(
            name=action_name,
            screen_pos=(screen_x, screen_y),
            scale=kwargs.get("scale", 1.0),
            delta=kwargs.get("delta", (0.0, 0.0)),
            extra=kwargs,
        )
        self.dispatch.dispatch(ev)

    def run(self):
        self._running = True
        print("[FingerController] Running - press Q in preview to quit.")
        fps_ts, fps_count = time.time(), 0

        while self._running:
            t0 = time.perf_counter()

            ret, frame = self.cap.read()
            if not ret:
                print("[FingerController] Camera read failed.")
                break

            self._frame_index += 1
            should_process = (self._frame_index % self._process_every_n_frames) == 0

            if should_process:
                fs_list, annotated = self.tracker.process(frame)
                if fs_list:
                    ev = self.gesture.update(fs_list)
                    if ev:
                        self.dispatch.dispatch(ev)
            else:
                annotated = self.tracker.transform_frame(frame)

            fps_count += 1
            now = time.time()
            if now - fps_ts >= 1.0:
                fps = fps_count / (now - fps_ts)
                fps_count = 0
                fps_ts = now
                if self.show_preview:
                    cv2.setWindowTitle("Finger Controller", f"Finger Controller  |  {fps:.0f} fps")

            if self.show_preview:
                cv2.imshow("Finger Controller", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            elapsed = time.perf_counter() - t0
            sleep = self._frame_dt - elapsed
            if sleep > 0:
                time.sleep(sleep)

        self.stop()

    def stop(self):
        self._running = False
        self.tracker.close()
        self.cap.release()
        if self.show_preview:
            cv2.destroyAllWindows()
        print("[FingerController] Stopped.")
