import pyautogui

from .types import ActionEvent
from .overlay import overlay

class ActionDispatcher:
    """Receives ActionEvents and performs screen operations via PyAutoGUI."""

    def __init__(self):
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0

    def dispatch(self, event: ActionEvent):
        x, y = event.screen_pos

        if event.name == "MOVE":
            pyautogui.moveTo(x, y, duration=0)

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
            # Scale comes back as a fraction like 0.2, 0.5, 1.2, etc. based on finger spread.
            steps = max(1, int(event.scale * 10))
            pyautogui.keyDown("ctrl")
            pyautogui.scroll(steps, x=x, y=y)
            pyautogui.keyUp("ctrl")
            overlay.show("Zoom In", 0.5)

        elif event.name == "ZOOM_OUT":
            # Scale comes back as a fraction indicating finger pinch distance.
            steps = max(1, int(event.scale * 10))
            pyautogui.keyDown("ctrl")
            pyautogui.scroll(-steps, x=x, y=y)
            pyautogui.keyUp("ctrl")
            overlay.show("Zoom Out", 0.5)

        elif event.name == "SWIPE_3F_LEFT":
            pyautogui.hotkey("alt", "tab")  # Go back (browser/folder)
            overlay.show("Alt + Tab", 1.5)
            print("[Action] 3-Finger Swipe LEFT triggered")

        elif event.name == "SWIPE_3F_RIGHT":
            pyautogui.hotkey("alt", "shift", "tab") # Go forward (browser/folder)
            overlay.show("Alt + Shift + Tab", 1.5)
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

