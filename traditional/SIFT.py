import cv2
import numpy as np

def localize_sift(screen_bgr, template_bgr):
    """
    SIFT based localization.
    Returns: matches [(x, y, w, h, conf)], result_img
    """
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    out = screen_bgr.copy()
    
    # 1️⃣ SIFT detector
    sift = cv2.SIFT_create(nfeatures=8000)
    kp1, des1 = sift.detectAndCompute(template_gray, None)
    kp2, des2 = sift.detectAndCompute(screen_gray, None)
    
    if des1 is None or des2 is None:
        return [], out
        
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)
            
    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        
        # RANSAC Homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
        
        if H is not None:
            # Confidence based on inlier count (heuristic)
            inliers = np.sum(mask)
            conf = min(1.0, inliers / 50.0)
            
            h, w = template_gray.shape
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            
            cv2.polylines(out, [np.int32(dst)], True, (0, 255, 0), 4, cv2.LINE_AA)
            
            x, y, w_box, h_box = cv2.boundingRect(np.int32(dst))
            return [(x, y, w_box, h_box, conf)], out
            
    return [], out

if __name__ == "__main__":
    # Load images
    img_original = cv2.imread("img2.png")
    img_patch = cv2.imread("img2CA.jpeg")
    
    if img_original is not None and img_patch is not None:
        matches, res = localize_sift(img_original, img_patch)
        if matches:
            print(f"Match found with confidence: {matches[0][4]}")
            cv2.imshow("Detected Location", res)
            cv2.waitKey(0)
        else:
            print("No match found.")