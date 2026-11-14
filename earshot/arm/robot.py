"""Connection to the SO-101 follower through LeRobot.

Two ways to open it: `open_limp` for hand-guiding (motors never switched on) and `open_holding`
for motion (motors switched on at the pose the arm is already in, so nothing jumps).
Joint angles are degrees in LeRobot's convention, which matches the URDF used in kin.py.
"""
from __future__ import annotations

import numpy as np

from earshot import config

GRIPPER = "gripper"


def _make(max_step_deg: float):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    return SO101Follower(SO101FollowerConfig(port=config.ARM_PORT, id=config.ARM_ID,
                                             max_relative_target=max_step_deg))


def open_limp():
    """Connect with every motor off, for moving the arm by hand. Returns the robot."""
    robot = _make(5.0)
    robot.bus.connect()
    robot.bus.disable_torque()
    return robot


def open_holding(max_step_deg: float = 4.0):
    """Connect and hold the current pose. `max_step_deg` caps how far one command can move a joint."""
    robot = _make(max_step_deg)
    robot.bus.connect()
    robot.bus.disable_torque()
    here = robot.bus.sync_read("Present_Position", normalize=False)
    robot.bus.sync_write("Goal_Position", {m: int(v) for m, v in here.items()}, normalize=False)
    robot.configure()            # LeRobot's own motor settings; ends with the motors on, holding `here`
    return robot


def read_joints(robot) -> np.ndarray:
    """The five arm joint angles in degrees."""
    pos = robot.bus.sync_read("Present_Position", config.ARM_JOINTS)
    return np.array([pos[j] for j in config.ARM_JOINTS], dtype=float)


def read_gripper(robot) -> float:
    """Gripper opening, 0 (closed) to 100 (open)."""
    return float(robot.bus.sync_read("Present_Position", [GRIPPER])[GRIPPER])


def close(robot, limp: bool = True) -> None:
    """Disconnect. With `limp` the motors are switched off first: support the arm by hand."""
    robot.bus.disconnect(disable_torque=limp)
