import cv2
import sys
import logging
import pyautogui

# Suppress warnings from SuperGlue if any
logging.getLogger().setLevel(logging.ERROR)

from screen_locator import localize_screen as localize_screen_tm
from screen_matching import localize_screen as localize_screen_sg
from traditional import localize_sift, localize_orb, localize_ecc

pairs = [
    ("test_data/img2.png", "test_data/img2CA.jpeg"),
]

SHOW_VISUALIZATION = True  # Set to True to see side-by-side results

print("=== Performance & Confidence Comparison ===\n")
header = f"{'Image Pair':<32} | {'TM':<11} | {'SG':<11} | {'SIFT':<11} | {'ORB':<11} | {'ECC':<11}"
print(header)
print("-" * len(header))

# Redirect stdout to suppress debug prints from the underlying functions
import os
import sys
from contextlib import redirect_stdout
import time

# Create img directory if it doesn't exist
os.makedirs("img", exist_ok=True)

for screen_file, template_file in pairs:
    screen_bgr = cv2.imread(screen_file)
    template_bgr = cv2.imread(template_file)
    
    if screen_bgr is None or template_bgr is None:
        print(f"{screen_file} + {template_file:<20} | {'File missing':<20}")
        continue
        
    results = {}
    
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        # 1. Template Matching
        t0 = time.perf_counter()
        tm_matches, tm_img = localize_screen_tm(screen_bgr, template_bgr, match_threshold=0.3)
        tm_time = (time.perf_counter() - t0) * 1000
        results['TM'] = (tm_matches[0][4] if tm_matches else 0.0, tm_time, tm_img)
        
        # 2. SuperGlue
        t1 = time.perf_counter()
        sg_matches, sg_img = localize_screen_sg(screen_bgr, template_bgr)
        sg_time = (time.perf_counter() - t1) * 1000
        results['SG'] = (sg_matches[0][4] if sg_matches else 0.0, sg_time, sg_img)

        # 3. SIFT
        t2 = time.perf_counter()
        sift_matches, sift_img = localize_sift(screen_bgr, template_bgr)
        sift_time = (time.perf_counter() - t2) * 1000
        results['SIFT'] = (sift_matches[0][4] if sift_matches else 0.0, sift_time, sift_img)

        # 4. ORB
        t3 = time.perf_counter()
        orb_matches, orb_img = localize_orb(screen_bgr, template_bgr)
        orb_time = (time.perf_counter() - t3) * 1000
        results['ORB'] = (orb_matches[0][4] if orb_matches else 0.0, orb_time, orb_img)

        # 5. ECC
        t4 = time.perf_counter()
        ecc_matches, ecc_img = localize_ecc(screen_bgr, template_bgr)
        ecc_time = (time.perf_counter() - t4) * 1000
        results['ECC'] = (ecc_matches[0][4] if ecc_matches else 0.0, ecc_time, ecc_img)
    
    pair_str = f"{screen_file} / {template_file}"
    if len(pair_str) > 32:
        pair_str = pair_str[:29] + "..."
    
    row = f"{pair_str:<32}"
    for method in ['TM', 'SG', 'SIFT', 'ORB', 'ECC']:
        conf, msec, _ = results[method]
        row += f" | {conf:.2f}/{msec:3.0f}ms"
    print(row)

    if SHOW_VISUALIZATION:
        # Show comparison windows
        sw, sh = pyautogui.size()
        target_w = int(sw * 0.3)
        target_h = int(sh * 0.3)

        for i, method in enumerate(['TM', 'SG', 'SIFT', 'ORB', 'ECC']):
            win_name = f"{method} - {screen_file}"
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, target_w, target_h)
            # Position windows in a grid if possible
            row_idx = i // 3
            col_idx = i % 3
            cv2.moveWindow(win_name, col_idx * (target_w + 10), row_idx * (target_h + 40))
            cv2.imshow(win_name, results[method][2])
            
            # Save the visualization to img/ folder
            base_name = os.path.basename(screen_file)
            save_path = os.path.join("img", f"{method}_{base_name}")
            cv2.imwrite(save_path, results[method][2])
        
        print(f"   -> Visualizations saved to img/ directory.")
        print("   -> Press any key to see next pair...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print("-" * len(header))
