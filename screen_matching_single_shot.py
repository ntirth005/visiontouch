import cv2
import numpy as np
from mss import mss
import time
import torch

from SuperGlueSuperPoint.models.matching import Matching
from SuperGlueSuperPoint.models.utils import frame2tensor

torch.set_grad_enabled(False)

# ----------------------------
# CONFIG
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 8192  # 8192 is completely safe for RAM, but gives way more points than 2048
    },
    'superglue': {
        'weights': 'outdoor',
        'sinkhorn_iterations': 20,
        'match_threshold': 0.05, # Lower threshold slightly to allow more valid matches
    }
}

matching = Matching(config).eval().to(device)

WINDOW_LEFT = 1400
WINDOW_TOP = 0
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 400

def print_screen_map(x_min, y_min, x_max, y_max, screen_w=1920, screen_h=1200, cols=64, rows=24):
    RESET     = "\033[0m"
    BG_SCREEN = "\033[48;5;234m"
    BG_MATCH  = "\033[48;5;28m"
    BORDER    = "\033[48;5;226m"
 
    def sc(v, src, dst): return v * dst / src
 
    bx0 = sc(x_min, screen_w, cols)
    by0 = sc(y_min, screen_h, rows)
    bx1 = sc(x_max, screen_w, cols)
    by1 = sc(y_max, screen_h, rows)
 
    for row in range(rows):
        line = ""
        for col in range(cols):
            in_x = bx0 <= col < bx1
            in_y = by0 <= row < by1
            if in_x and in_y:
                on_edge = (col == int(bx0) or col >= int(bx1) - 1 or
                           row == int(by0) or row >= int(by1) - 1)
                line += (BORDER + " " + RESET) if on_edge else (BG_MATCH + " " + RESET)
            else:
                line += BG_SCREEN + " " + RESET
        print(line)


def preprocess_gray(img):
    return img


def capture_screen_snapshot(sct):
    monitor = {
        "top": 0,
        "left": 0,
        "width": 1400,
        "height": 1200
    }
    frame = np.array(sct.grab(monitor))
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def open_live_window(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.moveWindow(window_name, WINDOW_LEFT, WINDOW_TOP)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)


# ----------------------------
# SUPERGLUE DETECTION FUNCTION
# ----------------------------
def detect_with_superglue(screen_bgr, template_gray):
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)

    template_gray = preprocess_gray(template_gray)
    screen_gray = preprocess_gray(screen_gray)

    template_tensor = frame2tensor(template_gray, device)
    screen_tensor = frame2tensor(screen_gray, device)

    # Extract template features
    data0 = matching.superpoint({'image': template_tensor})
    data0 = {k+'0': data0[k] for k in ['keypoints','scores','descriptors']}
    data0['image0'] = template_tensor

    # Match
    pred = matching({**data0, 'image1': screen_tensor})

    kpts0 = data0['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    scores = pred['matching_scores0'][0].cpu().numpy()
    mconf = scores[valid]
    mean_conf = np.mean(mconf) if len(mconf) > 0 else 0.0

    # print(f"Debug: found {len(kpts0)} keypoints in template, {len(kpts1)} in screen.")
    # print(f"Debug: valid superglue matches = {len(mkpts0)}")

    out = screen_bgr.copy()

    if len(mkpts0) >= 4:
        # Use USAC_MAGSAC instead of RANSAC. It is significantly more robust 
        # and creates stable bounding boxes even with few or clustered points.
        H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 5.0)

        if H is not None:
            h, w = template_gray.shape

            corners = np.float32([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ]).reshape(-1, 1, 2)

            projected = cv2.perspectiveTransform(corners, H)

            # Draw accurate polygon
            cv2.polylines(out, [np.int32(projected)], True, (0, 255, 0), 3)

            # Bounding box
            xs = projected[:, 0, 0]
            ys = projected[:, 0, 1]
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())

            cv2.rectangle(out, (x_min, y_min), (x_max, y_max), (255, 0, 0), 5)

            # Draw the actual matched points so you can see them!
            for pt in mkpts1:
                cv2.circle(out, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

            print(f"Matches: {len(mkpts0)} (Conf: {mean_conf:.3f})")

            return [(x_min, y_min, x_max-x_min, y_max-y_min, float(mean_conf))], out

    print("Not enough matches / failed")
    return [], out


# ----------------------------
# MAIN (same structure as yours)
# ----------------------------
def main():
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError("Failed to open webcam")

    sct = mss()
    window_name = "Live Screen Box"
    preview_open = False

    template_gray = None
    screen_result = None

    while True:
        if template_gray is None:
            if not preview_open:
                open_live_window(window_name)
                preview_open = True

            ok, frame = cap.read()
            if not ok:
                continue

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 32:  # SPACE → select ROI
                roi = cv2.selectROI(window_name, frame, False)
                x, y, w, h = [int(v) for v in roi]

                template_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

                cv2.destroyWindow(window_name)
                preview_open = False
                time.sleep(0.3)

                screen_bgr = capture_screen_snapshot(sct)

                matches, screen_result = detect_with_superglue(screen_bgr, template_gray)

            elif key in (27, ord('q')):
                break

            continue

        if not preview_open:
            open_live_window(window_name)
            preview_open = True

        cv2.imshow(window_name, screen_result)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            template_gray = None
            screen_result = None
        elif key == ord('q'):
            break

    cap.release()
    if preview_open:
        cv2.destroyWindow(window_name)


if __name__ == '__main__':
    main()