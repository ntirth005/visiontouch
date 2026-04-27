import time

import cv2
import pyautogui

from .calibration import calibrate_from_frame
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
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self._frame_dt = 1.0 / float(target_fps)
        self._process_every_n_frames = max(1, int(process_every_n_frames))
        self._frame_index = 0
        self.camera_rotation_deg = normalize_rotation_deg(camera_rotation_deg)
        self.mirror_input = mirror_input

        screen_w, screen_h = pyautogui.size()
        print(f"[FingerController] Screen: {screen_w}x{screen_h}")

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera (index={camera_index}). "
                "Try a different camera_index or check camera permissions."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # We capture at high resolution to get the full FOV, but we will 
        # downscale internally for processing to keep CPU usage low.
        self.target_process_width = 640
        cam_w_raw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h_raw = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        scale = self.target_process_width / cam_w_raw
        cam_w = int(cam_w_raw * scale)
        cam_h = int(cam_h_raw * scale)
        
        print(f"[FingerController] Camera Raw: {cam_w_raw}x{cam_h_raw} -> Process: {cam_w}x{cam_h}")

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
        self._is_calibrated = False

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
        print("[FingerController] Running - press SPACE to calibrate to screen, Q to quit.")
        fps_ts, fps_count = time.time(), 0

        try:
            while self._running:
                t0 = time.perf_counter()

                ret, frame = self.cap.read()
                if not ret:
                    print("[FingerController] Camera read failed.")
                    break

                # Downscale for performance while keeping the 720p FOV
                h_orig, w_orig = frame.shape[:2]
                if w_orig > self.target_process_width:
                    scale = self.target_process_width / w_orig
                    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                self._frame_index += 1
                should_process = (self._frame_index % self._process_every_n_frames) == 0

                if should_process:
                    fs_list, annotated = self.tracker.process(frame)
                    if fs_list and self._is_calibrated:
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
                    if self.show_preview and hasattr(cv2, "setWindowTitle"):
                        cv2.setWindowTitle(
                            "Finger Controller", f"Finger Controller  |  {fps:.0f} fps"
                        )

                if self.show_preview:
                    if not self._is_calibrated:
                        cv2.putText(annotated, "Press SPACE to Calibrate", (10, 60), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                    cv2.imshow("Finger Controller", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == 32:  # Spacebar
                        print("[FingerController] Calibrating...")
                        bbox, screen_result = calibrate_from_frame(frame)
                        if bbox:
                            x, y, w, h = bbox
                            self.mapper.screen_x = x
                            self.mapper.screen_y = y
                            self.mapper.screen_w = w
                            self.mapper.screen_h = h
                            self._is_calibrated = True
                            cv2.imshow("Live Screen Box Calibration", screen_result)
                            cv2.waitKey(1000)
                            cv2.destroyWindow("Live Screen Box Calibration")

                elapsed = time.perf_counter() - t0
                sleep = self._frame_dt - elapsed
                if sleep > 0:
                    time.sleep(sleep)
        finally:
            self.stop()

    def stop(self):
        self._running = False
        self.tracker.close()
        self.cap.release()
        if self.show_preview:
            cv2.destroyAllWindows()
        print("[FingerController] Stopped.")
