import sys
import os
import cv2
import numpy as np

from mss import mss
import sys
import os

# Add parent directory to sys.path in case this is imported differently
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from screen_matching import capture_screen, localize_screen, print_screen_map
# from screen_locator import capture_screen, localize_screen, print_screen_map

# We use mss via context manager inside the function to avoid leaks.

def calibrate_from_frame(frame: np.ndarray, match_threshold: float = 0.6) -> tuple[tuple[int, int, int, int] | None, np.ndarray]:
    """
    Captures the current screen and uses the webcam frame as a template to find 
    the screen bounds visible in the camera.
    Returns:
        (bbox, screen_result): bbox is (x, y, w, h) of the best match, or None if no match.
                               screen_result is the annotated screen snapshot.
    """
    with mss() as sct:
        screen_bgr = capture_screen(sct)
        
        # Use SuperGlue for calibration
        matches, screen_result = localize_screen(screen_bgr, frame)
    
    if matches:
        # Matches are sorted by confidence descending, so we take the first one.
        best_match = matches[0]
        x, y, w, h, conf = best_match
        print(f"[Calibration] Best match found: {w}x{h} at {x},{y} (conf: {conf:.2f})")
        
        # Print the ASCII screen map for terminal feedback
        sh, sw = screen_bgr.shape[:2]
        print_screen_map(x, y, x + w, y + h, screen_w=sw, screen_h=sh)
        
        return (x, y, w, h), screen_result
    else:
        print("[Calibration] No match found.")
        return None, screen_result
