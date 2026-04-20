import cv2
import torch

from models.utils import frame2tensor
from models.matching import Matching

def capture_template_features(
	matching,
	device,
	camera_index=1,
	window_name="Capture Template",
	window_pos=(20, 20),
	keep_on_top=True,
	close_on_capture=False,
):
	cap = cv2.VideoCapture(camera_index)

	if not cap.isOpened():
		raise RuntimeError(f"Failed to open webcam index {camera_index}")

	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.moveWindow(window_name, window_pos[0], window_pos[1])
	if keep_on_top:
		cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

	print("Press SPACE to capture template")

	small = None
	while True:
		ret, frame_cam = cap.read()
		if not ret:
			continue

		cv2.imshow(window_name, frame_cam)
		key = cv2.waitKey(1) & 0xFF

		if key == 32:
			small = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2GRAY)
			break

		if key == 27:
			cap.release()
			cv2.destroyWindow(window_name)
			raise RuntimeError("Template capture cancelled (ESC pressed)")

		if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
			cap.release()
			raise RuntimeError("Template capture cancelled (window closed)")

	if close_on_capture:
		cv2.destroyWindow(window_name)

	small_tensor = frame2tensor(small, device)
	with torch.no_grad():
		pred = matching.superpoint({'image': small_tensor})

	kpts0 = pred['keypoints'][0]
	desc0 = pred['descriptors'][0]
	scores0 = pred['scores'][0]

	return cap, small, small_tensor, kpts0, desc0, scores0

matching = Matching({
    'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 256},
    'superglue': {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}
}).eval().to('cpu')

capture_template_features(matching=matching, device='cpu', camera_index=1, window_name="Capture Template")