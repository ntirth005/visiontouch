import cv2
import numpy as np
from mss import mss
import time


WINDOW_LEFT = 1400
WINDOW_TOP = 0
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 400


def preprocess_gray(img):
    # Equalize contrast to reduce lighting mismatch between webcam and screen.
    return cv2.equalizeHist(img)


def non_max_suppression_rects(rects, scores, overlap_thresh=0.3):
    if len(rects) == 0:
        return []

    rects_np = np.array(rects, dtype=np.float32)
    x1 = rects_np[:, 0]
    y1 = rects_np[:, 1]
    x2 = rects_np[:, 0] + rects_np[:, 2]
    y2 = rects_np[:, 1] + rects_np[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / areas[order[1:]]

        remaining = np.where(overlap <= overlap_thresh)[0]
        order = order[remaining + 1]

    return keep


def capture_screen_snapshot(sct):
    # if len(sct.monitors) > 1:
    #     monitor = sct.monitors[1]
    # else:
    #     monitor = sct.monitors[0]
    # print(f"Using monitor: {monitor}")

    monitor = {
        "top": 0,
        "left": 0,
        "width": 1400,
        "height": 1200
    }
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def open_live_window(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.moveWindow(window_name, WINDOW_LEFT, WINDOW_TOP)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)


def detect_template_on_screen(screen_bgr, template_gray, match_threshold=0.70):
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    screen_gray = preprocess_gray(screen_gray)
    out = screen_bgr.copy()

    base_h, base_w = template_gray.shape
    if screen_gray.shape[0] < 20 or screen_gray.shape[1] < 20 or base_h < 20 or base_w < 20:
        return out

    template_gray = preprocess_gray(template_gray)
    template_edge_base = cv2.Canny(template_gray, 60, 150)

    # Multi-scale search improves robustness when webcam template size differs on-screen.
    scales = [0.60, 0.70, 0.80, 0.90, 1.00, 1.15, 1.30, 1.45]
    candidates = []

    for scale in scales:
        t_w = int(base_w * scale)
        t_h = int(base_h * scale)
        if t_w < 20 or t_h < 20:
            continue
        if t_h > screen_gray.shape[0] or t_w > screen_gray.shape[1]:
            continue

        tpl = cv2.resize(template_gray, (t_w, t_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        tpl_edge = cv2.resize(template_edge_base, (t_w, t_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)

        res_gray = cv2.matchTemplate(screen_gray, tpl, cv2.TM_CCOEFF_NORMED)
        screen_edge = cv2.Canny(screen_gray, 60, 150)
        res_edge = cv2.matchTemplate(screen_edge, tpl_edge, cv2.TM_CCOEFF_NORMED)

        # Blend texture + edge evidence to reduce false positives.
        combined = 0.65 * res_gray + 0.35 * res_edge

        ys, xs = np.where(combined >= match_threshold)
        for x, y in zip(xs, ys):
            candidates.append((int(x), int(y), int(t_w), int(t_h), float(combined[y, x])))

    if not candidates:
        # Fallback to strongest match at native scale for guaranteed output.
        result = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        x, y = max_loc
        print(f"Threshold: {match_threshold:.2f}")
        print("Matches above threshold: 0")
        print(f"Best match only: conf={max_val:.3f}, top_left=({x}, {y}), bottom_right=({x + base_w}, {y + base_h})")
        cv2.rectangle(out, (x, y), (x + base_w, y + base_h), (0, 0, 255), 2)
        label = f"best {max_val:.2f} ({x},{y})"
        cv2.putText(out, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        return out

    rects = [(x, y, w, h) for x, y, w, h, _ in candidates]
    scores = [s for _, _, _, _, s in candidates]
    keep_idxs = non_max_suppression_rects(rects, scores, overlap_thresh=0.2)

    matches = []
    for idx in keep_idxs:
        x, y, w, h = rects[idx]
        conf = scores[idx]
        matches.append((x, y, w, h, conf))

    matches.sort(key=lambda m: m[4], reverse=True)
    matches = matches[:12]

    print(f"Threshold: {match_threshold:.2f}")
    print(f"Matches above threshold: {len(matches)}")
    for i, (x, y, w, h, conf) in enumerate(matches, start=1):
        print(f"Match {i}: conf={conf:.3f}, top_left=({x}, {y}), bottom_right=({x + w}, {y + h})")

    for x, y, w, h, conf in matches:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
        label = f"{conf:.2f} ({x},{y})"
        cv2.putText(out, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    return out


def main():
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open webcam index {camera_index}")

    sct = mss()
    window_name = "Live Screen Box"
    preview_open = False

    template_gray = None
    match_threshold = 0.60
    screen_result = None

    while True:
        if template_gray is None:
            if not preview_open:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
                cv2.moveWindow(window_name, WINDOW_LEFT, WINDOW_TOP)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                preview_open = True

            ok, frame = cap.read()
            if not ok:
                continue

            view = frame.copy()
            cv2.imshow(window_name, view)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                # Select object ROI from webcam frame for much better matching accuracy.
                roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
                x, y, w, h = [int(v) for v in roi]
                if w > 10 and h > 10:
                    template_gray = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
                else:
                    template_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Close webcam preview before taking screenshot so it is not captured.
                cv2.destroyWindow(window_name)
                preview_open = False
                time.sleep(0.35)

                screen_bgr = capture_screen_snapshot(sct)
                screen_result = detect_template_on_screen(screen_bgr, template_gray, match_threshold=match_threshold)
            elif key in (27, ord('q')):
                break
            continue

        if not preview_open:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
            cv2.moveWindow(window_name, WINDOW_LEFT, WINDOW_TOP)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
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