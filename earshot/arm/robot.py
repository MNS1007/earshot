"""Connection to the SO-101 follower through LeRobot.

Two ways to open it: `open_limp` for hand-guiding (motors never switched on) and `open_holding`
for motion (motors switched on at the pose the arm is already in, so nothing jumps).

Joint angles: LeRobot reports degrees measured from the middle of the range each joint was
swept through at calibration. The URDF in kin.py measures from the calibration pose itself.
The two differ by a fixed amount per joint (up to 19 degrees on this arm), worked out here
from the saved calibration, so `read_joints` and `to_motor` speak the URDF's angles.
"""
from __future__ import annotations

import numpy as np

from earshot import config

GRIPPER = "gripper"


def _make(max_step_deg: float):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    # LeRobot cuts any command that asks a joint to be further than this from where it is now.
    # The wrist roll and the gripper are light and travel far, so they get more room; without
    # it the gripper stopped half closed and the wrist arrived late.
    limit = {j: max_step_deg for j in config.ARM_JOINTS}
    limit["wrist_roll"] = 4 * max_step_deg
    limit[GRIPPER] = 100.0
    return SO101Follower(SO101FollowerConfig(port=config.ARM_PORT, id=config.ARM_ID, max_relative_target=limit))


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


def joint_offsets(robot) -> np.ndarray:
    """Degrees to add to LeRobot's angle to get the URDF's angle, per arm joint."""
    out = []
    for j in config.ARM_JOINTS:
        cal = robot.bus.calibration[j]
        max_res = robot.bus.resolution(robot.bus.motors[j].model) - 1
        half_turn = max_res // 2                      # the calibration pose reads exactly this
        out.append(((cal.range_min + cal.range_max) / 2 - half_turn) * 360.0 / max_res + config.ARM_ZERO_FIX.get(j, 0.0))
    return np.array(out)


def read_joints(robot) -> np.ndarray:
    """The five arm joint angles in degrees, in the URDF's convention."""
    pos = robot.bus.sync_read("Present_Position", config.ARM_JOINTS)
    return np.array([pos[j] for j in config.ARM_JOINTS], dtype=float) + joint_offsets(robot)


def to_motor(robot, joints_deg) -> np.ndarray:
    """URDF-convention angles -> the degrees LeRobot expects in a command."""
    return np.asarray(joints_deg, dtype=float) - joint_offsets(robot)


def read_gripper(robot) -> float:
    """Gripper opening, 0 (closed) to 100 (open)."""
    return float(robot.bus.sync_read("Present_Position", [GRIPPER])[GRIPPER])


def close(robot, limp: bool = True) -> None:
    """Disconnect. With `limp` the motors are switched off first: support the arm by hand."""
    robot.bus.disconnect(disable_torque=limp)
