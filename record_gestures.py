"""
record_gestures.py
──────────────────
Records raw finger landmark data from your webcam into a CSV file.
Compatible with MediaPipe 0.10+  (uses the new Tasks API).

Setup:
    pip install opencv-python mediapipe

    # Download the hand landmarker model (one-time):
    curl -o hand_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

    # Then run:
    python record_gestures.py

Controls (in the webcam window):
    SPACE  — start / stop recording the current label
    1-9    — switch gesture label before recording
    q      — quit and save

Output:
    gesture_log.csv   — one row per frame with all finger distances +
                        which fingers are "up", tagged with your gesture label.

After quitting a summary table is printed in the terminal showing the
min/max px range of each key distance per label — copy those numbers
directly into gestures.py as your thresholds.
"""

import csv
import math
import time
import urllib.request
from pathlib import Path
from collections import defaultdict

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_FILE   = "gesture_log.csv"
CAMERA_INDEX  = 0
MODEL_PATH    = "hand_landmarker.task"
MODEL_URL     = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
FINGER_UP_THRESHOLD = 0.05   # normalised units; tip must be this far above MCP

# MediaPipe landmark indices
TIP   = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
MCP   = {"thumb": 2, "index": 5, "middle":  9, "ring": 13, "pinky": 17}
WRIST = 0

LABELS = {
    "1": "move_index",
    "2": "move_middle",
    "3": "pinch_close",
    "4": "pinch_open",
    "5": "zoom_apart",
    "6": "zoom_together",
    "7": "fist",
    "8": "all_fingers",
    "9": "other",
}

# ── Auto-download model if missing ────────────────────────────────────────────

def ensure_model():
    p = Path(MODEL_PATH)
    if not p.exists():
        print(f"Downloading hand landmarker model → {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  Done.\n")

# ── Measurement helpers ───────────────────────────────────────────────────────

def dist_norm(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def dist_px(a, b, w: int, h: int) -> float:
    return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)


def is_up(lm, finger: str) -> bool:
    tip_y = lm[TIP[finger]].y
    mcp_y = lm[MCP[finger]].y
    return (mcp_y - tip_y) > FINGER_UP_THRESHOLD


def extract_row(lm, label: str, w: int, h: int) -> dict:
    thumb_tip  = lm[TIP["thumb"]]
    index_tip  = lm[TIP["index"]]
    middle_tip = lm[TIP["middle"]]
    wrist      = lm[WRIST]

    return {
        "ts":                   round(time.time(), 4),
        "label":                label,
        # Fingers up
        "thumb_up":             int(is_up(lm, "thumb")),
        "index_up":             int(is_up(lm, "index")),
        "middle_up":            int(is_up(lm, "middle")),
        "ring_up":              int(is_up(lm, "ring")),
        "pinky_up":             int(is_up(lm, "pinky")),
        # Distances (normalised)
        "pinch_dist_norm":      round(dist_norm(thumb_tip, index_tip), 4),
        "two_finger_dist_norm": round(dist_norm(index_tip, middle_tip), 4),
        # Distances (pixels) — use these for threshold tuning
        "pinch_dist_px":        round(dist_px(thumb_tip, index_tip, w, h), 1),
        "two_finger_dist_px":   round(dist_px(index_tip, middle_tip, w, h), 1),
        # Fingertip positions (normalised)
        "index_x":              round(index_tip.x, 4),
        "index_y":              round(index_tip.y, 4),
        "middle_x":             round(middle_tip.x, 4),
        "middle_y":             round(middle_tip.y, 4),
        # Hand scale proxy (wrist→middle MCP px) for normalising across distances
        "hand_scale_px":        round(dist_px(wrist, lm[MCP["middle"]], w, h), 1),
    }

# ── Draw landmarks manually (Tasks API doesn't bundle a draw util) ────────────

_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS  # still accessible as a constant

def draw_landmarks(frame, landmarks, w: int, h: int):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (80, 200, 80), 1)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 220, 0), -1)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_model()

    # Build landmarker in VIDEO mode (synchronous, per-frame)
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = mp_vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    recording   = False
    label       = "move_index"
    rows        = []
    frame_count = 0
    frame_ts_ms = 0

    print(f"\nRecorder ready.  Output → {OUTPUT_FILE}")
    print("Keys: SPACE=start/stop  1-9=set label  q=quit\n")
    for k, v in LABELS.items():
        print(f"  {k} → {v}")
    print()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame      = cv2.flip(frame, 1)
        h, w       = frame.shape[:2]
        frame_ts_ms += 33   # approximate; Tasks API needs monotonic ms

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        result = landmarker.detect_for_video(mp_image, frame_ts_ms)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]   # list of NormalizedLandmark
            draw_landmarks(frame, lm, w, h)

            if recording:
                rows.append(extract_row(lm, label, w, h))
                frame_count += 1

        # HUD
        status_color = (0, 200, 0) if recording else (0, 120, 220)
        status_text  = (
            f"REC [{label}]  {frame_count} frames"
            if recording
            else f"PAUSED  total={len(rows)}"
        )
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, "SPACE=rec  1-9=label  q=quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Gesture Recorder", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            recording   = not recording
            frame_count = 0
            print(f"  {'RECORDING' if recording else 'STOPPED'} → label={label}")
        elif chr(key) in LABELS:
            label = LABELS[chr(key)]
            print(f"  Label → {label}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    if rows:
        out = Path(OUTPUT_FILE)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {len(rows)} frames → {out.resolve()}")
        _print_summary(rows)
    else:
        print("\nNo data recorded.")


def _print_summary(rows: list[dict]):
    buckets: dict[str, list] = defaultdict(list)
    for r in rows:
        buckets[r["label"]].append(r)

    cols = ["pinch_dist_px", "two_finger_dist_px", "hand_scale_px"]
    col_w = 26

    print("\n── Summary (px distances) " + "─" * 50)
    print(f"{'label':<22}", end="")
    for c in cols:
        print(f"  {c:<{col_w}}", end="")
    print()
    print("─" * (22 + len(cols) * (col_w + 2)))

    for lbl, bucket in sorted(buckets.items()):
        print(f"{lbl:<22}", end="")
        for col in cols:
            vals = [float(r[col]) for r in bucket]
            mn, mx = min(vals), max(vals)
            print(f"  {mn:6.1f} – {mx:6.1f}{'':12}", end="")
        print(f"  ({len(bucket)} frames)")

    print()
    print("→ Use these ranges to set thresholds in gestures.py:")
    print("    PINCH_CLOSE_PX        ← max of pinch_close  pinch_dist_px")
    print("    PINCH_OPEN_PX         ← min of pinch_open   pinch_dist_px")
    print("    ZOOM_DEAD_ZONE_PX     ← ~half the jitter range in zoom_apart/together")


if __name__ == "__main__":
    main()
