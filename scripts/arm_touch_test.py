"""Measure where the fixed jaw really lands, in the pose used for picking. THE ARM MOVES.

Clear the sheet. For three calibration marks the arm opens the gripper as it does for a pick,
turns the wrist the same way, and lowers the fixed jaw's tip onto the mark, then a camera
picture is saved. The tip is on the table in those pictures, so the gap between tip and mark
can be read straight off them and turned into a correction.

    python scripts/arm_touch_test.py
"""
import json

import cv2
import numpy as np

from earshot import config
from earshot.arm import kin, motion
from earshot.camera import Camera

cam = Camera().start()
bot = motion.Arm(speed=0.35)
out = config.DATA / "touch_test"
out.mkdir(parents=True, exist_ok=True)
marks = json.loads(config.CALIB_FILE.read_text())["marks"]
try:
    print(bot.follow_camera(cam.fresh_frame()))
    input("Sheet clear, marks 2, 3 and 4 uncovered? Press Enter: the arm will touch each of them...")
    bot.set_gripper(kin.gripper_for_gap(motion.RAKE_GAP), wait=0.2)
    for n in (3, 4, 2):
        x, y = marks[n - 1]["arm_xy"]
        u, v = bot.table.to_pixel(x, y)
        away = np.array(bot.table.to_arm(u, v - 12.0)) - np.array([x, y])
        bot.jaw_deg = float(np.rad2deg(np.arctan2(away[1], away[0])))
        try:
            bot.move_xyz(x, y, motion.ABOVE_TABLE_HOVER)
            bot.move_height(motion.ABOVE_TABLE_APPROACH)
            bot.lower_until_touch()
        except motion.ReachError as e:
            print(f"mark {n}: {e}")
            continue
        frame = cam.fresh_frame()
        cv2.imwrite(str(out / f"mark_{n}.jpg"), frame)
        print(f"mark {n}: aimed at x {x * 100:.1f} cm, y {y * 100:.1f} cm = pixel ({u:.0f}, {v:.0f}); "
              f"jaw line {bot.jaw_deg:.0f} deg; tip measured {bot.tip_height() * 1000:.0f} mm above the table; "
              f"joints {np.round(motion.arm.read_joints(bot.robot))}")
        bot.move_height(motion.ABOVE_TABLE_HOVER)
    bot.jaw_deg = None
finally:
    bot.close()
    cam.stop()
    print("Motors off. Pictures are in", out)
