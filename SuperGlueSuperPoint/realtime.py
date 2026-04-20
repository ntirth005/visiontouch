import cv2
import torch
import numpy as np
import time
from mss import mss

from models.matching import Matching
from models.utils import frame2tensor
from webcame import capture_template_features

# ---------------- DEVICE ----------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- MODEL ----------------
matching = Matching({
    'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 256},
    'superglue': {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}
}).eval().to(device)

# ---------------- WEBCAM ----------------
cap, small, small_tensor, kpts0, desc0, scores0 = capture_template_features(
    matching=matching,
    device=device,
    camera_index=1,
    window_name="Capture Template",
    window_pos=(20, 20),
    keep_on_top=True,
    close_on_capture=False,
)

# ---------------- SCREEN ----------------
sct = mss()
monitor = {
    "top": 0,
    "left": 0,
    "width": 1200,
    "height": 1160
}
print(sct.monitors)
# ---------------- TIMER ----------------
last_run = 0
interval = 2
last_box = None

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame_cam = cap.read()
    if not ret:
        continue

    # GREEN REGION (webcam)
    green = cv2.resize(frame_cam, (400, 240))

    # RED REGION (screen)
    frame_screen = np.array(sct.grab(monitor))
    frame_screen = cv2.cvtColor(frame_screen, cv2.COLOR_BGRA2BGR)
    
    frame_screen = cv2.resize(frame_screen, (800, 600))
    red = frame_screen.copy()

    frame_gray = cv2.cvtColor(frame_screen, cv2.COLOR_BGR2GRAY)

    current_time = time.time()

    # MATCH EVERY 2 SECONDS
    if current_time - last_run > interval:

        frame_tensor = frame2tensor(frame_gray, device)

        with torch.no_grad():
            pred1 = matching.superpoint({'image': frame_tensor})

        data = {
            'keypoints0': kpts0.unsqueeze(0),
            'keypoints1': pred1['keypoints'][0].unsqueeze(0),
            'descriptors0': desc0.unsqueeze(0),
            'descriptors1': pred1['descriptors'][0].unsqueeze(0),
            'scores0': scores0.unsqueeze(0),
            'scores1': pred1['scores'][0].unsqueeze(0),
            'image0': small_tensor,
            'image1': frame_tensor,
        }

        with torch.no_grad():
            pred = matching.superglue(data)

        matches = pred['matches0'][0].cpu().numpy()
        conf = pred['matching_scores0'][0].cpu().numpy()

        valid = (matches > -1) & (conf > 0.5)

        if np.sum(valid) > 10:
            mkpts0 = kpts0.cpu().numpy()[valid]
            mkpts1 = pred1['keypoints'][0].cpu().numpy()[matches[valid]]

            M, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = small.shape
                pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, M)

                last_box = dst.astype(int)

        last_run = current_time

    # YELLOW REGION (result)
    yellow = red.copy()
    if last_box is not None:
        cv2.polylines(yellow, [last_box], True, (0,255,255), 2)

    yellow = cv2.resize(yellow, (400, 240))

    # ---------------- COMBINE UI ----------------
    top = red
    bottom = np.hstack((green, yellow))
    ui = np.vstack((top, bottom))

    cv2.imshow("System View", ui)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()