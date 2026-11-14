"""Central settings: model names, paths, ports, and table geometry.

Everything that might change at the venue lives here so no other file hard-codes it.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"                      # git-ignored: clips, photos, calibration, saved tasks
TASKS_DIR = DATA / "tasks"
CALIB_FILE = DATA / "camera_to_arm.json"

# Models (Gemini API), chosen per job from timed runs on the same clips and photo.
WATCH_MODEL = "gemini-4-argon"        # clip + words -> steps, and applying rules: right on both clips, about 27 s
POINT_MODEL = "gemini-3.8-flash"      # finds items and checks placements: fastest (about 2.5 s), points at the base
POINT_MODEL_ALT = "gemini-4-argon"    # about 5 s, also points at the base
GEMINI_BASE = "https://generativelanguage.googleapis.com"

# Camera
CAMERA_NAME = "HD Pro Webcam C920"
CAMERA_SIZE = (1280, 720)

# Arm
ARM_PORT = "/dev/tty.usbmodem5B790502351"
ARM_ID = "follower"
URDF_PATH = str(Path.home() / "developer/SO-ARM100/Simulation/SO101/so101_new_calib.urdf")
EE_FRAME = "gripper_frame_link"
ARM_CALIBRATION = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower" / f"{ARM_ID}.json"
# Extra degrees between this arm's motor zero and the URDF's, beyond what the calibration file
# explains. None is needed. Measured on 2026-11-14 with scripts/arm_roll_test.py: with the wrist
# motor at -90, 0 and +90 the swinging jaw pointed right, away from the base and left, which is
# what the URDF predicts. (Two earlier guesses of +90 and -90 here were wrong.)
ARM_ZERO_FIX: dict[str, float] = {}
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Heights above the table, in metres, in the arm's base frame (z of the table found at calibration).
HOVER_Z = 0.06        # travel height (straight down is only reachable up to about here)
GRASP_Z = 0.015       # fingertip height when closing on an item lying flat
DROP_Z = 0.14         # release height over a cup (the gripper leans at this height)

# Where the arm sets an item down for the person to take, in the arm's frame (metres).
HANDOVER_XY = (0.15, 0.19)

# Local services
UI_PORT = 8765        # screen + state API
VOICE_POST_URL = f"http://127.0.0.1:{UI_PORT}/api/voice"   # the voice process posts actions here


def key(name: str) -> str:
    """Read an API key from the environment or the git-ignored .env file."""
    if name in os.environ:
        return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError(f"{name} is not set. Put it in {env} (never commit that file).")
