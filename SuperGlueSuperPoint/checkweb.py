from webcame import capture_template_features
from models.matching import Matching

matching = Matching({
    'superpoint': {'nms_radius': 4, 'keypoint_threshold': 0.005, 'max_keypoints': 256},
    'superglue': {'weights': 'outdoor', 'sinkhorn_iterations': 20, 'match_threshold': 0.2}
}).eval().to('cpu')

capture_template_features(matching=matching, device='cpu', camera_index=1, window_name="Capture Template")