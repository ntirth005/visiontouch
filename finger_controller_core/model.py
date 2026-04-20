import os
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "hand_landmarker.task",
)


def ensure_model() -> None:
    if not os.path.exists(MODEL_PATH):
        print("[FingerController] Downloading hand landmark model (~8 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[FingerController] Model saved -> {MODEL_PATH}")
