"""Entry point for the finger controller app."""

import os
import sys

# Set environment variables BEFORE any imports (important for TensorFlow/MediaPipe)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

from finger_controller_core import FingerController

if __name__ == "__main__":
    ctrl = FingerController(
        camera_index=1,
        show_preview=True,
        target_fps=25,
        process_every_n_frames=1,
        camera_rotation_deg=0,
        mirror_input=False,
    )
    ctrl.run()
    