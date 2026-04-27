import os
import urllib.request
import urllib.error

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "hand_landmarker.task",
)


def ensure_model() -> None:
    if not os.path.exists(MODEL_PATH):
        print("[FingerController] Downloading hand landmark model (~8 MB)...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                "Failed to download MediaPipe hand landmark model. "
                "If you're offline, download it manually from:\n"
                f"  {MODEL_URL}\n"
                "and save it to:\n"
                f"  {MODEL_PATH}"
            ) from exc
        print(f"[FingerController] Model saved -> {MODEL_PATH}")
