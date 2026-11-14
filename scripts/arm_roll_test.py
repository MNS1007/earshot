"""Measure which way the jaws face for a given wrist-roll motor angle. THE ARM MOVES (slowly, in the air).

Clear the acrylic sheet first. The arm goes to a point 7 cm above the middle of the sheet,
gripper pointing down and opened, then turns the wrist to three motor angles and saves a
camera picture at each. Looking at where the swinging jaw points in each picture gives the
real relation between the motor's roll angle and the jaw direction, instead of guessing it.

    python scripts/arm_roll_test.py
"""
import cv2
import numpy as np

from earshot import config
from earshot.arm import kin, motion, robot as arm
from earshot.camera import Camera

cam = Camera().start()
bot = motion.Arm(speed=0.3)
out = config.DATA / "roll_test"
out.mkdir(parents=True, exist_ok=True)
try:
    input("Sheet clear? Press Enter: the arm will hover over the middle of the sheet and turn its wrist...")
    x, y = 0.20, 0.06
    q, err, lean, ok = kin.ik_down(x, y, bot.table.table_z + 0.07)
    print(f"hover pose joints {np.round(q, 1)} (error {err * 1000:.1f} mm, lean {lean:.0f} deg)")
    fix = arm.joint_offsets(bot.robot)[4]
    for motor_roll in (-90.0, 0.0, 90.0):
        target = q.copy()
        target[4] = motor_roll + fix              # so that the motor itself is asked for motor_roll
        bot.move_joints(target)
        bot.set_gripper(kin.gripper_for_gap(0.085))
        frame = cam.fresh_frame()
        cv2.imwrite(str(out / f"motor_roll_{int(motor_roll):+04d}.jpg"), frame)
        print(f"motor roll {motor_roll:+.0f}: picture saved, pan joint {target[0]:.0f}")
    bot.set_gripper(10.0)
finally:
    bot.close()
    cam.stop()
    print("Motors off. Pictures are in", out)
