from typing import Optional

import pyautogui

from .types import ActionEvent
from .overlay import overlay

# Attempt to load Windows-specific touch injection for native pinch-to-zoom.
# This is optional and should never prevent the app from starting.
HAS_TOUCH = False
try:
    from .windows_touch import touch_injector

    HAS_TOUCH = True
    print("[System] Touch injector module loaded (lazy init; may fall back to Ctrl+Scroll).")
except Exception:
    HAS_TOUCH = False

class ActionDispatcher:
    """Receives ActionEvents and performs screen operations via PyAutoGUI."""

    def __init__(self):
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        # Cursor smoothing (reduces micro-jitter/flicker)
        self._cursor_smooth_alpha = 0.35  # 0..1 (higher = snappier)
        self._cursor_deadzone_px = 2.0    # ignore tiny moves
        self._cursor_sx: Optional[float] = None
        self._cursor_sy: Optional[float] = None
        self._cursor_last_sent: tuple[int, int] = (0, 0)

        self.is_pinching = False
        self.pinch_x = 0
        self.pinch_y = 0
        self.pinch_accumulated_scale = 0.0

    def dispatch(self, event: ActionEvent):
        x, y = event.screen_pos

        # Only end an active pinch on events that genuinely conflict.
        # MOVE events are emitted by the gesture engine during dead-zone /
        # grace-period frames and must NOT kill the pinch session.
        _PINCH_SAFE = {"ZOOM_IN", "ZOOM_OUT", "MOVE"}
        if event.name not in _PINCH_SAFE:
            if getattr(self, 'is_pinching', False):
                self.is_pinching = False
                if HAS_TOUCH:
                    print("[Action] ENDING Continuous Pinch")
                    touch_injector.end_pinch(self.pinch_x, self.pinch_y)

        if event.name == "MOVE":
            if self._cursor_sx is None or self._cursor_sy is None:
                self._cursor_sx = float(x)
                self._cursor_sy = float(y)
                self._cursor_last_sent = (x, y)
                pyautogui.moveTo(x, y, duration=0)
            else:
                a = self._cursor_smooth_alpha
                self._cursor_sx = (1.0 - a) * self._cursor_sx + a * float(x)
                self._cursor_sy = (1.0 - a) * self._cursor_sy + a * float(y)
                tx = int(self._cursor_sx)
                ty = int(self._cursor_sy)
                lx, ly = self._cursor_last_sent
                if abs(tx - lx) >= self._cursor_deadzone_px or abs(ty - ly) >= self._cursor_deadzone_px:
                    self._cursor_last_sent = (tx, ty)
                    pyautogui.moveTo(tx, ty, duration=0)

        elif event.name == "CLICK":
            pyautogui.click(x, y)
            overlay.show("Click", 0.5)

        elif event.name == "RIGHT_CLICK":
            pyautogui.rightClick(x, y)
            overlay.show("Right Click", 0.5)

        elif event.name == "DOUBLE_CLICK":
            pyautogui.doubleClick(x, y)
            overlay.show("Double Click", 0.5)

        elif event.name == "SCROLL_UP":
            pyautogui.scroll(3, x=x, y=y)

        elif event.name == "SCROLL_DOWN":
            pyautogui.scroll(-3, x=x, y=y)

        elif event.name == "SCROLL_LEFT":
            pyautogui.hscroll(-3, x=x, y=y)

        elif event.name == "SCROLL_RIGHT":
            pyautogui.hscroll(3, x=x, y=y)

        elif event.name == "ZOOM_IN":
            if HAS_TOUCH:
                if not self.is_pinching:
                    self.is_pinching = True
                    self.pinch_x, self.pinch_y = x, y
                    self.pinch_accumulated_scale = 100.0
                    print("[Action] STARTING Continuous Pinch")
                    touch_injector.start_pinch(self.pinch_x, self.pinch_y, initial_dist=100)
                else:
                    # Only update on subsequent frames (not the start frame)
                    self.pinch_accumulated_scale += (event.scale * 10)
                    new_dist = max(20, self.pinch_accumulated_scale)
                    touch_injector.update_pinch(self.pinch_x, self.pinch_y, new_dist)
            else:
                steps = max(1, int(event.scale * 20))
                import ctypes
                import time
                ctypes.windll.user32.SetCursorPos(x, y)
                time.sleep(0.02)
                pyautogui.keyDown("ctrl")
                pyautogui.scroll(steps)
                pyautogui.keyUp("ctrl")
            overlay.show("Zoom In", 0.5)

        elif event.name == "ZOOM_OUT":
            if HAS_TOUCH:
                if not self.is_pinching:
                    self.is_pinching = True
                    self.pinch_x, self.pinch_y = x, y
                    self.pinch_accumulated_scale = 100.0
                    print("[Action] STARTING Continuous Pinch")
                    touch_injector.start_pinch(self.pinch_x, self.pinch_y, initial_dist=100)
                else:
                    # Only update on subsequent frames (not the start frame)
                    self.pinch_accumulated_scale -= (event.scale * 10)
                    new_dist = max(20, self.pinch_accumulated_scale)
                    touch_injector.update_pinch(self.pinch_x, self.pinch_y, new_dist)
            else:
                steps = max(1, int(event.scale * 20))
                import ctypes
                import time
                ctypes.windll.user32.SetCursorPos(x, y)
                time.sleep(0.02)
                pyautogui.keyDown("ctrl")
                pyautogui.scroll(-steps)
                pyautogui.keyUp("ctrl")
            overlay.show("Zoom Out", 0.5)

        elif event.name == "ZOOM_END":
            if getattr(self, 'is_pinching', False):
                self.is_pinching = False
                if HAS_TOUCH:
                    print("[Action] ENDING Continuous Pinch")
                    touch_injector.end_pinch(self.pinch_x, self.pinch_y)

        elif event.name == "SWIPE_3F_LEFT":
            pyautogui.hotkey("alt", "tab")  
            overlay.show("Previous", 1.0)
            print("[Action] 3-Finger Swipe LEFT triggered")

        elif event.name == "SWIPE_3F_RIGHT":
            pyautogui.hotkey("alt", "shift", "tab")
            overlay.show("Next", 1.0)
            print("[Action] 3-Finger Swipe RIGHT triggered")

        elif event.name == "SWIPE_4F_LEFT":
            # Activated to the previous working approach:
            pyautogui.hotkey("ctrl", "win", "left") # Switch virtual desktop
            
            # Alternative if 'hotkey' drops the win key in the future:
            # pyautogui.keyDown("ctrl")
            # pyautogui.keyDown("win")
            # pyautogui.press("left")
            # pyautogui.keyUp("win")
            # pyautogui.keyUp("ctrl")
            
            overlay.show("Desktop Left", 1.5)
            print("[Action] 4-Finger Swipe LEFT triggered")  

        elif event.name == "SWIPE_4F_RIGHT":
            # Activated to the previous working approach:
            pyautogui.hotkey("ctrl", "win", "right") # Switch virtual desktop
            
            # Alternative if 'hotkey' drops the win key in the future:
            # pyautogui.keyDown("ctrl")
            # pyautogui.keyDown("win")
            # pyautogui.press("right")
            # pyautogui.keyUp("win")
            # pyautogui.keyUp("ctrl")
            
            overlay.show("Desktop Right", 1.5)
            print("[Action] 4-Finger Swipe RIGHT triggered")

