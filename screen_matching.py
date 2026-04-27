import cv2
import numpy as np
import torch
from mss import mss
import time

from SuperGlueSuperPoint.models.matching import Matching
from SuperGlueSuperPoint.models.utils import frame2tensor

torch.set_grad_enabled(False)


# CONFIG

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'superpoint': {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': 2048  # With downscaling, 2048 is more than enough
    },
    'superglue': {
        'weights': 'outdoor',  # 'outdoor' is generally more robust
        'sinkhorn_iterations': 20,
        'match_threshold': 0.05,
    }
}

matching = Matching(config).eval().to(device)


# SCREEN CAPTURE & UTILS

def capture_screen(sct=None):
    if sct is None:
        with mss() as s_sct:
            return _do_capture(s_sct)
    else:
        return _do_capture(sct)

def _do_capture(sct):
    # sct.monitors[0] is the bounding box of all monitors
    # sct.monitors[1] is the first (primary) monitor
    monitor = sct.monitors[1]
    img = np.array(sct.grab(monitor))
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def print_screen_map(x_min, y_min, x_max, y_max,
                     screen_w=1920, screen_h=1200,
                     cols=40, rows=12):

    def sc(v, src, dst):
        return int((v / src) * dst)

    # Clamp
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(screen_w, x_max)
    y_max = min(screen_h, y_max)

    # Scale
    bx0 = sc(x_min, screen_w, cols)
    by0 = sc(y_min, screen_h, rows)
    bx1 = sc(x_max, screen_w, cols)
    by1 = sc(y_max, screen_h, rows)

    for row in range(rows):
        line = ""
        for col in range(cols):

            # === BIG BOX ===
            if row == 0 and col == 0:
                line += "┌"
            elif row == 0 and col == cols-1:
                line += "┐"
            elif row == rows-1 and col == 0:
                line += "└"
            elif row == rows-1 and col == cols-1:
                line += "┘"
            elif row == 0 or row == rows-1:
                line += "─"
            elif col == 0 or col == cols-1:
                line += "│"

            # === SMALL BOX ===
            elif bx0 <= col < bx1 and by0 <= row < by1:

                if row == by0 and col == bx0:
                    line += "┌"
                elif row == by0 and col == bx1-1:
                    line += "┐"
                elif row == by1-1 and col == bx0:
                    line += "└"
                elif row == by1-1 and col == bx1-1:
                    line += "┘"
                elif row == by0 or row == by1-1:
                    line += "─"
                elif col == bx0 or col == bx1-1:
                    line += "│"
                else:
                    line += " "

            else:
                line += " "

        print(line)

def resize_image_max_dim(img, max_dim=640):
    """Resizes an image so its longest side is max_dim, returning the resized image and scale factor."""
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img, 1.0
    
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def localize_screen(screen_bgr, template_bgr):
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)

    # Downscale for speed
    template_small, scale_template = resize_image_max_dim(template_gray, 640)
    screen_small, scale_screen = resize_image_max_dim(screen_gray, 640)

    template_tensor = frame2tensor(template_small, device)
    screen_tensor = frame2tensor(screen_small, device)

    # Features
    with torch.no_grad():
        data0 = matching.superpoint({'image': template_tensor})
        data0 = {k+'0': data0[k] for k in ['keypoints','scores','descriptors']}
        data0['image0'] = template_tensor

        pred = matching({**data0, 'image1': screen_tensor})

    kpts0 = data0['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    scores = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]
    mconf = scores[valid]

    if len(mkpts0) < 4:
        print(f"[SuperGlue] Failed: Not enough raw matches ({len(mkpts0)})")
        return [], screen_bgr

    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, 5.0)
    if H is None:
        print("[SuperGlue] Failed: Homography could not be estimated.")
        return [], screen_bgr

    inliers = np.sum(mask) if mask is not None else 0
    if inliers < 4:
        print(f"[SuperGlue] Failed: Not enough inliers ({inliers} / {len(mkpts0)})")
        return [], screen_bgr

    mean_conf = np.mean(mconf) if len(mconf) > 0 else 0.0
    h_small, w_small = template_small.shape
    corners = np.float32([[0,0],[w_small,0],[w_small,h_small],[0,h_small]]).reshape(-1,1,2)
    projected_small = cv2.perspectiveTransform(corners, H)
    projected_full = projected_small / scale_screen

    xs, ys = projected_full[:, 0, 0], projected_full[:, 0, 1]
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    # Draw visualization on a copy of the screen
    out = screen_bgr.copy()
    cv2.rectangle(out, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

    return [(x_min, y_min, x_max-x_min, y_max-y_min, float(mean_conf))], out


# MAIN

def main():
    cap = cv2.VideoCapture(0)

    print("Press SPACE to select the full webcam frame as the template.")

    template_data = None
    template_small = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE
            print("Capturing full webcam frame as template...")
            template = frame.copy()
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            # --- NEW RESIZE LOGIC ---
            template_small, scale_template = resize_image_max_dim(template_gray, 640)
            template_tensor = frame2tensor(template_small, device)


            data0 = matching.superpoint({'image': template_tensor})
            template_data = {k+'0': data0[k] for k in ['keypoints','scores','descriptors']}
            template_data['image0'] = template_tensor

            print("Template selected. Capturing screen...")
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cv2.destroyWindow("Webcam")
    cv2.waitKey(1)  # Process the destroy window event
    time.sleep(0.4) # Wait for the window fade animation to finish
    cap.release()

    
    # SINGLE SHOT CAPTURE & MATCH
    
    screen = capture_screen()
    t_start = time.perf_counter()
    matches, _ = localize_screen(screen, template)
    t_end = time.perf_counter()
    duration_ms = (t_end - t_start) * 1000

    if matches:
        x, y, w, h, mean_conf = matches[0]
        x_min, y_min, x_max, y_max = x, y, x + w, y + h
        
        # Visualize on full res screen
        cv2.rectangle(screen, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

        print(f"\n--- Result (SuperGlue) ---")
        print(f"Confidence      : {mean_conf:.3f}")
        print(f"Time Taken      : {duration_ms:.1f} ms")
        print(f"Position (X,Y)  : {x_min}, {y_min}")
        print(f"Size (WxH)      : {w} x {h}\n")

        print_screen_map(x_min, y_min, x_max, y_max)

    else:
        print("Failed! Not enough matches.")

    
    # DISPLAY
    
    cv2.namedWindow("Detection Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Detection Result", screen)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()