import numpy as np
import cv2

# load results
data = np.load('output\\vscodeC_vscode_matches.npz')

kpts0 = data['keypoints0']
kpts1 = data['keypoints1']
matches = data['matches']

# filter valid matches
valid = matches > -1
matched_kpts0 = kpts0[valid]
matched_kpts1 = kpts1[matches[valid]]

# load images
small = cv2.imread('images\\vscodeC.png')
large = cv2.imread('images\\vscode.png')

# compute homography
M, mask = cv2.findHomography(
    matched_kpts0,
    matched_kpts1,
    cv2.RANSAC,
    5.0
)

# get bounding box
h, w = small.shape[:2]
pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)

dst = cv2.perspectiveTransform(pts, M)

print("Coordinates:")
print(dst)

# draw box on large image
dst = dst.astype(int)

cv2.polylines(large, [dst], True, (0,255,0), 3)
# get screen size (approx safe values)
screen_width = 1200
screen_height = 700

h_img, w_img = large.shape[:2]

# compute scaling factor
scale = min(screen_width / w_img, screen_height / h_img)

# resize only if needed
if scale < 1:
    new_w = int(w_img * scale)
    new_h = int(h_img * scale)
    large = cv2.resize(large, (new_w, new_h))
    
cv2.imshow("Result", large)
cv2.waitKey(0)
cv2.destroyAllWindows()