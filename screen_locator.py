import cv2
import numpy as np
from mss import mss
import time


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

    
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


def capture_screen(sct):
    # sct.monitors[1] is the primary monitor
    monitor = sct.monitors[1]
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def open_live_window(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)


def localize_screen(screen_bgr, template_bgr, match_threshold=0.70):
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    
    screen_gray = preprocess_gray(screen_gray)
    template_gray = preprocess_gray(template_gray)
    out = screen_bgr.copy()

    base_h, base_w = template_gray.shape
    if screen_gray.shape[0] < 20 or screen_gray.shape[1] < 20 or base_h < 20 or base_w < 20:
        return [], out

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
        return [(x, y, base_w, base_h, float(max_val))], out

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

    for i, (x, y, w, h, conf) in enumerate(matches):
        # Draw the best match in red (0, 0, 255), others in cyan/yellow (0, 255, 255)
        color = (0, 0, 255) if i == 0 else (0, 255, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"{conf:.2f} ({x},{y})"
        cv2.putText(out, label, (x, max(15, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return matches, out


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

                t_start = time.perf_counter()
                screen_bgr = capture_screen(sct)
                matches, screen_result = localize_screen(screen_bgr, frame, match_threshold=match_threshold)
                t_end = time.perf_counter()
                duration_ms = (t_end - t_start) * 1000
                
                if matches:
                    x, y, w, h, conf = matches[0]
                    x_min, y_min, x_max, y_max = x, y, x + w, y + h
                    
                    print(f"\n--- Result (Template Matching) ---")
                    print(f"Confidence      : {conf:.3f}")
                    print(f"Time Taken      : {duration_ms:.1f} ms")
                    print(f"Position (X,Y)  : {x_min}, {y_min}")
                    print(f"Size (WxH)      : {w} x {h}\n")

                    print_screen_map(x_min, y_min, x_max, y_max)
            elif key in (27, ord('q')):
                break
            continue

        if not preview_open:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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