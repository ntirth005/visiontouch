import cv2
import numpy as np

def localize_ecc(screen_bgr, template_bgr):
    """
    ECC based localization with Template Matching initialization.
    Returns: matches [(x, y, w, h, conf)], result_img
    """
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    out = screen_bgr.copy()

    # 1. TM Initialization (ECC needs to be close to the target)
    res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    if max_val < 0.2:
        return [], out
        
    x0, y0 = max_loc
    h, w = template_gray.shape
    
    # 2. ECC Refinement
    # Crop the target area based on TM result to give ECC a good starting point
    # We expand the crop slightly to give ECC room to align
    pad = 10
    y1 = max(0, y0 - pad)
    y2 = min(screen_gray.shape[0], y0 + h + pad)
    x1 = max(0, x0 - pad)
    x2 = min(screen_gray.shape[1], x0 + w + pad)
    
    screen_crop = screen_gray[y1:y2, x1:x2]
    
    # We need to warp the template to match the screen crop
    warp_mode = cv2.MOTION_TRANSLATION
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    # Set termination criteria
    number_of_iterations = 500
    termination_eps = 1e-8
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    
    try:
        # Run ECC
        _, warp_matrix = cv2.findTransformECC(template_gray, screen_crop, warp_matrix, warp_mode, criteria)
        
        # Calculate final coordinates relative to the original screen
        # warp_matrix gives the translation of the template within the crop
        dx = warp_matrix[0, 2]
        dy = warp_matrix[1, 2]
        
        # Final bounding box
        final_x = int(x1 + dx)
        final_y = int(y1 + dy)
        
        cv2.rectangle(out, (final_x, final_y), (final_x + w, final_y + h), (0, 165, 255), 2) # Orange
        return [(final_x, final_y, w, h, float(max_val))], out
        
    except cv2.error:
        # If ECC fails to converge, fallback to the original TM result
        cv2.rectangle(out, (x0, y0), (x0 + w, y0 + h), (0, 165, 255), 2)
        return [(x0, y0, w, h, float(max_val))], out

if __name__ == "__main__":
    img_original = cv2.imread('img.png')
    img_patch = cv2.imread('imgC.png')
    if img_original is not None and img_patch is not None:
        matches, res = localize_ecc(img_original, img_patch)
        if matches:
            cv2.imshow("ECC Location", res)
            cv2.waitKey(0)
