"""Entry point for the finger controller app."""

import os
import sys

import logging
logging.getLogger('absl').setLevel(logging.ERROR)

from finger_controller_core import FingerController

if __name__ == "__main__":
    try:
        ctrl = FingerController(
            camera_index=0,
            show_preview=True,
            target_fps=25,
            process_every_n_frames=1,
            camera_rotation_deg=0,
            mirror_input=False,
        )
        ctrl.run()
    except Exception as exc:
        print(f"[FingerController] Fatal: {exc}")
        sys.exit(1)
