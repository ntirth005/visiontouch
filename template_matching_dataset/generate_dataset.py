"""
Template Matching Dataset Generator
====================================
Captures a webcam photo + screenshot simultaneously, detects the screen
region, corrects perspective, and produces augmented training samples
with tracked coordinates — all saved in a structured dataset format.

Usage
-----
  # Live capture (press ENTER → 3s countdown → capture)
  python generate_dataset.py --capture

  # With pre-saved images
  python generate_dataset.py --template screenshot.png --camera photo.jpg

  # Built-in demo (synthetic images, no camera needed)
  python generate_dataset.py --demo

Output structure:
  dataset_output/
  ├── img_0001/
  │   ├── original.png        ← perspective-corrected screen
  │   ├── aug_1.png … aug_20.png
  │   ├── aug_1_crop_0.png …  ← sub-image crops
  │   └── ...
  ├── img_0001.json           ← coordinates & metadata
  ├── img_0002/
  │   └── ...
  └── img_0002.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np


# ───────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────

def _next_img_index(output_dir: str) -> int:
    """Find the next available img_XXXX index in output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("img_")
    ]
    if not existing:
        return 1
    indices = []
    for d in existing:
        try:
            indices.append(int(d.split("_")[1]))
        except (ValueError, IndexError):
            pass
    return max(indices, default=0) + 1


def visual_crop_tkinter(image_bgr: np.ndarray, title: str = "Visual Crop") -> Optional[Tuple[int, int, int, int]]:
    """Shows a Tkinter window to draw a crop box, avoiding cv2.imshow which crashes on CI environments."""
    import tkinter as tk
    import tempfile
    
    # Save image to a temporary file that tkinter's PhotoImage can load natively (PNG)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    cv2.imwrite(tmp_path, image_bgr)
    
    root = tk.Tk()
    root.title(title)
    
    try:
        # Maximize window on Windows
        root.state('zoomed')
    except Exception:
        pass
        
    tk.Label(
        root, 
        text="Click and drag to select crop area. Press ENTER to confirm, or ESC to skip.", 
        bg="black", fg="white", font=("Arial", 12)
    ).pack(fill="x")
    
    img = tk.PhotoImage(file=tmp_path)
    canvas = tk.Canvas(root, width=img.width(), height=img.height(), cursor="cross")
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, anchor="nw", image=img)
    
    rect_id = None
    start_x = start_y = 0
    crop_coords = None

    def on_press(event):
        nonlocal start_x, start_y, rect_id
        start_x = canvas.canvasx(event.x)
        start_y = canvas.canvasy(event.y)
        if rect_id:
            canvas.delete(rect_id)
        rect_id = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline="red", width=2)

    def on_drag(event):
        cur_x = canvas.canvasx(event.x)
        cur_y = canvas.canvasy(event.y)
        canvas.coords(rect_id, start_x, start_y, cur_x, cur_y)

    def on_confirm(event):
        nonlocal crop_coords
        if rect_id:
            x1, y1, x2, y2 = canvas.coords(rect_id)
            if x1 != x2 and y1 != y2:
                crop_coords = (int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2)))
        root.quit()

    def on_cancel(event):
        root.quit()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    root.bind("<Return>", on_confirm)
    root.bind("<Escape>", on_cancel)

    root.mainloop()

    try:
        root.destroy()
    except tk.TclError:
        pass
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return crop_coords


# ───────────────────────────────────────────────────
# 1. LIVE CAPTURE  (terminal-based, no GUI needed)
# ───────────────────────────────────────────────────

def capture_images(output_dir: str, cam_index: int = 1) -> Tuple[str, str]:
    """
    Open webcam → user presses ENTER → 3-second countdown →
    simultaneously grab screenshot + webcam frame.
    Then ask the user to crop the screenshot to select the
    relevant screen region (the template).
    No cv2.imshow / no GUI window required.
    """
    try:
        import mss
    except ImportError:
        print("[!] 'mss' is required.  pip install mss")
        sys.exit(1)

    cap_dir = os.path.join(output_dir, "_captures")
    os.makedirs(cap_dir, exist_ok=True)

    # Try requested camera index, then fallback to 0
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check your camera connection.")

    # Request high resolution (1080p) from webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Warm-up (auto-exposure)
    print("[*] Warming up camera...")
    for _ in range(30):
        cap.read()

    print()
    print("╔═══════════════════════════════════════════════╗")
    print("║  LIVE CAPTURE                                 ║")
    print("║  Point camera at the screen, then press ENTER ║")
    print("║  A 3-second countdown will start.             ║")
    print("╚═══════════════════════════════════════════════╝")
    print()
    input(">>> Press ENTER when ready... ")

    for s in range(3, 0, -1):
        print(f"  Capturing in {s}...")
        time.sleep(1)
    print("  Capturing NOW!")

    # ── Screenshot (mss gives BGRA on Windows) ──
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        screenshot = np.array(raw)
        screenshot = screenshot[:, :, :3]  # BGRA → BGR

    # ── Webcam frame (grab a few to flush the buffer) ──
    for _ in range(5):
        ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Failed to read from webcam.")

    # Un-mirror the webcam image (webcams are often flipped horizontally)
    frame = cv2.flip(frame, 1)

    ts = int(time.time())
    ss_path = os.path.join(cap_dir, f"screenshot_{ts}.png")
    cam_path = os.path.join(cap_dir, f"camera_{ts}.png")
    cv2.imwrite(ss_path, screenshot)
    cv2.imwrite(cam_path, frame)

    print(f"\n[✓] Screenshot → {ss_path}")
    print(f"[✓] Camera     → {cam_path}")

    # ── Ask user to crop the camera visually (select relevant region) ──
    print("\n[*] Opening visual cropper...")
    print("    A window will pop up. Drag a rectangle to crop the WEBCAM IMAGE.")
    print("    Press ENTER to confirm, or ESC to skip and use the full image.")
    
    coords = visual_crop_tkinter(frame, title="Crop Webcam Image")

    if coords:
        x1, y1, x2, y2 = coords
        cropped_cam = frame[y1:y2, x1:x2]
        cropped_cam_path = os.path.join(cap_dir, f"camera_cropped_{ts}.png")
        cv2.imwrite(cropped_cam_path, cropped_cam)
        print(f"[✓] Cropped camera image ({x2-x1}x{y2-y1}) → {cropped_cam_path}")
        return ss_path, cropped_cam_path
    else:
        print("[!] No crop selected. Using full camera image.")

    return ss_path, cam_path


# ───────────────────────────────────────────────────
# 2. SCREEN DETECTION  &  PERSPECTIVE CORRECTION
# ───────────────────────────────────────────────────

def detect_screen_and_correct(
    template_path: str,
    camera_path: str,
) -> np.ndarray:
    """
    Detect the template inside the camera image and warp the camera image
    so the screen content is rectified to the template's dimensions.

    Uses SIFT (if available) with fallback to ORB.
    """
    template = cv2.imread(template_path)
    camera   = cv2.imread(camera_path)
    if template is None:
        raise FileNotFoundError(f"Cannot read template: {template_path}")
    if camera is None:
        raise FileNotFoundError(f"Cannot read camera image: {camera_path}")

    h_t, w_t = template.shape[:2]
    gray_t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_c = cv2.cvtColor(camera,   cv2.COLOR_BGR2GRAY)

    # ── Try SIFT first (better quality), fall back to ORB ──
    try:
        detector = cv2.SIFT_create(nfeatures=5000)
        norm = cv2.NORM_L2
        print("  Using SIFT detector")
    except cv2.error:
        detector = cv2.ORB_create(nfeatures=5000)
        norm = cv2.NORM_HAMMING
        print("  Using ORB detector (SIFT unavailable)")

    kp_t, des_t = detector.detectAndCompute(gray_t, None)
    kp_c, des_c = detector.detectAndCompute(gray_c, None)

    if des_t is None or des_c is None:
        raise RuntimeError("No features detected in one of the images.")
    print(f"  Features: template={len(kp_t)}, camera={len(kp_c)}")

    # ── Match with ratio test ──
    bf = cv2.BFMatcher(norm, crossCheck=False)
    raw = bf.knnMatch(des_t, des_c, k=2)

    good = []
    for pair in raw:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
    print(f"  Good matches: {len(good)}")

    if len(good) < 10:
        print(f"  [!] Only {len(good)} matches. Bypassing perspective correction.")
        print("      Falling back to direct resize of the camera image.")
        return cv2.resize(camera, (w_t, h_t), interpolation=cv2.INTER_CUBIC)

    # Points in template → points in camera
    pts_t = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_c = np.float32([kp_c[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Homography: maps camera coords → template coords
    H, mask = cv2.findHomography(pts_c, pts_t, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"  Homography inliers: {inliers}/{len(good)}")

    if H is None or inliers < 15:
        print("  [!] Homography inliers too low. Bypassing perspective correction.")
        print("      Falling back to direct resize of the camera image.")
        return cv2.resize(camera, (w_t, h_t), interpolation=cv2.INTER_CUBIC)

    corrected = cv2.warpPerspective(camera, H, (w_t, h_t))
    return corrected


# ───────────────────────────────────────────────────
# 3. AUGMENTATION  +  COORDINATE TRACKING
# ───────────────────────────────────────────────────

_TAG_MAP = {
    "Rotate": "rot", "HorizontalFlip": "hflip", "VerticalFlip": "vflip",
    "Affine": "affine", "RandomBrightnessContrast": "bright",
    "HueSaturationValue": "hue", "GaussNoise": "noise",
    "GaussianBlur": "blur", "Perspective": "persp",
    "RandomGamma": "gamma", "CLAHE": "clahe", "Sharpen": "sharp",
}


def _build_tag(replay: dict) -> str:
    tags = []
    for t in replay.get("transforms", []):
        name = t.get("__class_fullname__", "").split(".")[-1]
        if t.get("applied") and name in _TAG_MAP:
            tags.append(_TAG_MAP[name])
    return "_".join(tags) if tags else "base"


def _random_crops(
    h: int, w: int, n: int = 4,
    lo: float = 0.15, hi: float = 0.45,
) -> List[Tuple[int, int, int, int]]:
    boxes = []
    for _ in range(n):
        cw = random.randint(int(w * lo), int(w * hi))
        ch = random.randint(int(h * lo), int(h * hi))
        x1 = random.randint(0, w - cw)
        y1 = random.randint(0, h - ch)
        boxes.append((x1, y1, x1 + cw, y1 + ch))
    return boxes


# ── Distinct colors for up to 10 crops ──
_CROP_COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 165, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (0, 128, 255),
    (255, 128, 0), (128, 255, 0),
]


def _draw_verify_image(
    template_img: np.ndarray,
    original_kps: list,
    aug_fname: str,
    crop_list: list,
    save_path: str,
) -> None:
    """Draw original keypoints on a copy of the TEMPLATE so coordinates
    can be visually verified against the template image."""
    vis = template_img.copy()

    # Title: which augmented image this verify belongs to
    cv2.putText(vis, f"Verify: {aug_fname}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw original keypoints on the template
    for ki, kp in enumerate(original_kps):
        kx, ky = float(kp[0]), float(kp[1])
        pt = (int(kx), int(ky))
        cv2.circle(vis, pt, 8, (0, 0, 255), -1)
        cv2.circle(vis, pt, 8, (255, 255, 255), 2)
        cv2.putText(vis, f"kp{ki} ({int(kx)},{int(ky)})",
                    (pt[0] + 10, pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Draw crop regions (from augmented coords, shown for reference)
    for ci, crop in enumerate(crop_list):
        color = _CROP_COLORS[ci % len(_CROP_COLORS)]
        x1, y1, x2, y2 = crop["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = crop["file"]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    cv2.imwrite(save_path, vis)


def generate_augmented_dataset(
    template_img: np.ndarray,
    corrected: np.ndarray,
    keypoints: List[Tuple[float, float]],
    img_dir: str,
    json_path: str,
    n_aug: int = 20,
    n_crops: int = 4,
    verify: bool = False,
) -> dict:
    """
    Structure:
      img_XXXX/
      ├── org_img.png                ← corrected original
      └── aug/
          ├── img_1.png              ← full augmented
          ├── img_1-1.png … img_1-4  ← crops of augmented 1
          ├── img_1_verify.png       ← verification on template
          └── ...
    """
    aug_dir = os.path.join(img_dir, "aug")
    os.makedirs(aug_dir, exist_ok=True)
    h, w = corrected.shape[:2]
    base_dir = os.path.dirname(json_path)
    img_name = os.path.splitext(os.path.basename(json_path))[0]

    # Save corrected original inside img_dir
    cv2.imwrite(os.path.join(img_dir, "org_img.png"), corrected)
    print(f"[✓] Original  → {os.path.join(img_dir, 'org_img.png')}")

    # Save the template (mss screenshot) for reference
    cv2.imwrite(os.path.join(img_dir, "template.png"), template_img)
    print(f"[✓] Template  → {os.path.join(img_dir, 'template.png')}")

    # Ground truth verify on template (only if --verify)
    verify_fname = None
    if verify:
        verify_fname = f"{img_name}_verify.png"
        vis = template_img.copy()
        for ki, (kx, ky) in enumerate(keypoints):
            pt = (int(kx), int(ky))
            cv2.circle(vis, pt, 8, (0, 0, 255), -1)
            cv2.circle(vis, pt, 8, (255, 255, 255), 2)
            cv2.putText(vis, f"kp{ki}", (pt[0] + 10, pt[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(base_dir, verify_fname), vis)
        print(f"[✓] Ground truth verify → {os.path.join(base_dir, verify_fname)}")

    # Albumentations pipeline
    transform = A.ReplayCompose(
        [
            A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_REFLECT_101),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.1),
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5,
            ),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.Perspective(scale=(0.02, 0.06), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=2.0, p=0.2),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=0.2),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    metadata = {
        "image_width": w,
        "image_height": h,
        "original_file": "org_img.png",
        "original_keypoints": keypoints,
        "verify_file": verify_fname,
        "augmented": [],
    }

    for i in range(1, n_aug + 1):
        result = transform(image=corrected, keypoints=keypoints)
        aug_img = result["image"]
        aug_kps = [list(kp) for kp in result["keypoints"]]
        tag = _build_tag(result["replay"])

        # Full augmented image: aug/img_1.png
        aug_fname = f"img_{i}.png"
        cv2.imwrite(os.path.join(aug_dir, aug_fname), aug_img)

        # Sub-image crops: aug/img_1-1.png, img_1-2.png, ...
        ah, aw = aug_img.shape[:2]
        boxes = _random_crops(ah, aw, n=n_crops)
        crop_list = []
        for ci, (x1, y1, x2, y2) in enumerate(boxes):
            crop_fname = f"img_{i}-{ci + 1}.png"
            cv2.imwrite(os.path.join(aug_dir, crop_fname), aug_img[y1:y2, x1:x2])

            kps_in = []
            for ki, (kx, ky) in enumerate(aug_kps):
                if x1 <= kx <= x2 and y1 <= ky <= y2:
                    kps_in.append({
                        "kp_index": ki,
                        "x": round(kx - x1, 2),
                        "y": round(ky - y1, 2),
                    })

            crop_list.append({
                "file": f"aug/{crop_fname}",
                "bbox": [x1, y1, x2, y2],
                "w": x2 - x1, "h": y2 - y1,
                "keypoints": kps_in,
            })

        metadata["augmented"].append({
            "file": f"aug/{aug_fname}",
            "index": i,
            "tag": tag,
            "keypoints": aug_kps,
            "crops": crop_list,
        })

        # Verify image on template (only if enabled)
        if verify:
            _draw_verify_image(
                template_img, keypoints, aug_fname, crop_list,
                os.path.join(aug_dir, f"img_{i}_verify.png"),
            )
        print(f"  [{i:2d}/{n_aug}] img_{i}.png  tag={tag}  crops={len(crop_list)}")

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[✓] Metadata → {json_path}")

    return metadata


# ───────────────────────────────────────────────────
# 4. DEMO MODE  (synthetic test images)
# ───────────────────────────────────────────────────

def _create_demo_images(tmp_dir: str):
    os.makedirs(tmp_dir, exist_ok=True)

    # Template 640×480
    t = np.zeros((480, 640, 3), dtype=np.uint8)
    t[:] = (40, 40, 40)
    cv2.rectangle(t, (50, 50), (590, 430), (0, 180, 255), 3)
    cv2.circle(t, (320, 240), 80, (255, 100, 50), -1)
    cv2.putText(t, "TEMPLATE", (180, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.rectangle(t, (100, 350), (250, 410), (50, 200, 50), -1)
    cv2.rectangle(t, (400, 350), (550, 410), (200, 50, 50), -1)
    tp = os.path.join(tmp_dir, "demo_template.png")
    cv2.imwrite(tp, t)

    # Camera: warped template on noisy background
    bg = np.random.randint(30, 80, (720, 1280, 3), dtype=np.uint8)
    src = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
    dst = np.float32([[250, 100], [900, 140], [950, 560], [200, 580]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(t, M, (1280, 720))
    mask = cv2.warpPerspective(np.ones_like(t) * 255, M, (1280, 720))
    cam = np.where(mask > 0, warped, bg)
    cp = os.path.join(tmp_dir, "demo_camera.png")
    cv2.imwrite(cp, cam)

    kps = [
        (100., 350.), (250., 350.), (250., 410.), (100., 410.),
        (400., 350.), (550., 350.), (550., 410.), (400., 410.),
        (320., 240.),
    ]
    print(f"[demo] Template  → {tp}")
    print(f"[demo] Camera    → {cp}")
    return tp, cp, kps


# ───────────────────────────────────────────────────
# 5. CLI
# ───────────────────────────────────────────────────

def parse_keypoints(s: str):
    kps = []
    for p in s.split(";"):
        p = p.strip()
        if not p:
            continue
        x, y = p.split(",")
        kps.append((float(x), float(y)))
    return kps


def main():
    ap = argparse.ArgumentParser(
        description="Template-Matching Dataset Generator",
    )
    ap.add_argument("--capture", action="store_true",
                    help="Live capture: webcam + screenshot via countdown.")
    ap.add_argument("--cam-index", type=int, default=1,
                    help="Webcam device index (default: 1).")
    ap.add_argument("--template", type=str,
                    help="Path to template screenshot (if not using --capture).")
    ap.add_argument("--camera", type=str,
                    help="Path to camera photo (if not using --capture).")
    ap.add_argument("--keypoints", type=str, default="",
                    help='e.g. "100,200;300,400"')
    ap.add_argument("--output", type=str, default="dataset",
                    help="Root output directory.")
    ap.add_argument("--n-aug", type=int, default=20,
                    help="Augmented images per capture (default 20).")
    ap.add_argument("--n-crops", type=int, default=4,
                    help="Random crops per augmented image (default 4).")
    ap.add_argument("--verify", action="store_true",
                    help="Generate verification images (off by default).")
    ap.add_argument("--demo", action="store_true",
                    help="Run with synthetic images (no camera needed).")
    args = ap.parse_args()

    # ── Source images ──
    if args.demo:
        d = os.path.join(args.output, "_demo_inputs")
        templ, cam, kps = _create_demo_images(d)
    elif args.capture:
        templ, cam = capture_images(args.output, cam_index=args.cam_index)
        kps = parse_keypoints(args.keypoints) if args.keypoints else []
    else:
        if not args.template or not args.camera:
            ap.error("Provide --template + --camera, or use --capture / --demo.")
        templ, cam = args.template, args.camera
        kps = parse_keypoints(args.keypoints) if args.keypoints else []

    # ── Read template image for verification plotting ──
    template_img = cv2.imread(templ)

    # ── Perspective correction ──
    print("\n═══ Step 1: Detecting screen & correcting perspective ═══")
    corrected = detect_screen_and_correct(templ, cam)

    # ── Determine output folder name ──
    idx = _next_img_index(args.output)
    img_name = f"img_{idx:04d}"
    img_dir = os.path.join(args.output, img_name)
    json_path = os.path.join(args.output, f"{img_name}.json")

    # ── Augmentation ──
    print(f"\n═══ Step 2: Generating {args.n_aug} augmented images → {img_name}/ ═══\n")
    meta = generate_augmented_dataset(
        template_img, corrected, kps, img_dir, json_path,
        n_aug=args.n_aug, n_crops=args.n_crops, verify=args.verify,
    )

    total_crops = sum(len(a["crops"]) for a in meta["augmented"])
    print(f"\n═══ Done! ═══")
    print(f"  Folder          : {img_dir}")
    print(f"  Metadata        : {json_path}")
    print(f"  Augmented images: {len(meta['augmented'])}")
    print(f"  Sub-image crops : {total_crops}")
    print(f"  Keypoints       : {len(kps)}")


if __name__ == "__main__":
    main()
