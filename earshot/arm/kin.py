"""Kinematics for top-down grasps with the SO-101.

Wraps LeRobot's placo-based RobotKinematics. The arm has five joints, so it cannot reach an
arbitrary pose; for picking off a table we only ask for a position plus "gripper pointing
straight down, turned to face away from the base", which five joints can do.

Frames: the URDF base frame. x forward from the base, y to the arm's left, z up.
Angles in degrees, in LeRobot's joint order (see config.ARM_JOINTS).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np

from earshot import config


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


@lru_cache(maxsize=1)
def _solver():
    from lerobot.model.kinematics import RobotKinematics

    return RobotKinematics(config.URDF_PATH, target_frame_name=config.EE_FRAME, joint_names=config.ARM_JOINTS)


@lru_cache(maxsize=1)
def joint_limits_deg() -> dict[str, tuple[float, float]]:
    """Joint limits from the URDF, in degrees."""
    out = {}
    for j in ET.parse(config.URDF_PATH).getroot().iter("joint"):
        lim = j.find("limit")
        if j.get("name") in config.ARM_JOINTS and lim is not None:
            out[j.get("name")] = (np.rad2deg(float(lim.get("lower"))), np.rad2deg(float(lim.get("upper"))))
    return out


def fk(joints_deg) -> np.ndarray:
    """4x4 pose of the gripper frame for the five arm joints."""
    return np.array(_solver().forward_kinematics(np.asarray(joints_deg, dtype=float)))


@lru_cache(maxsize=1)
def _down_rotation() -> np.ndarray:
    """Gripper orientation when it points straight down with the arm facing +x."""
    # At the all-zero pose the gripper points along +x; tipping that 90 degrees about y points it down.
    return _rot_y(90.0) @ fk(np.zeros(len(config.ARM_JOINTS)))[:3, :3]


def down_pose(x: float, y: float, z: float) -> np.ndarray:
    """Target pose: fingertips at (x, y, z), pointing down, turned toward the point."""
    T = np.eye(4)
    T[:3, :3] = _rot_z(np.rad2deg(np.arctan2(y, x))) @ _down_rotation()
    T[:3, 3] = (x, y, z)
    return T


def tilt_from_down_deg(T: np.ndarray) -> float:
    """Angle between the gripper's approach axis and straight down."""
    approach = T[:3, :3] @ np.linalg.inv(_down_rotation())[:, 2] * -1.0  # world direction the fingers point
    return float(np.rad2deg(np.arccos(np.clip(approach @ np.array([0, 0, -1.0]), -1, 1))))


def _solve(target: np.ndarray, seed_deg: np.ndarray, down_weight: float, iters: int):
    q = np.array(_solver().inverse_kinematics(seed_deg, target, position_weight=1.0,
                                              orientation_weight=down_weight, max_iters=iters))
    q = q[: len(config.ARM_JOINTS)]
    T = fk(q)
    return q, float(np.linalg.norm(T[:3, 3] - target[:3, 3])), tilt_from_down_deg(T)


def ik_down(x: float, y: float, z: float, seed_deg=None, iters: int = 200, max_err_m: float = 0.004):
    """Joint angles that put the fingertips at (x, y, z), pointing down when the arm can.

    Straight down is not reachable everywhere: above about 6 cm the wrist runs out of travel.
    So this first asks for straight down, and if the position misses by more than `max_err_m`
    it asks again with the orientation only loosely held, which lets the gripper lean.

    Returns (joints_deg, position_error_m, tilt_deg, within_limits).
    """
    pan = np.rad2deg(np.arctan2(y, x))
    if seed_deg is None:
        # Elbow-up starting guess facing the target, so the solver does not fold the wrong way.
        seed_deg = np.array([pan, -20.0, 40.0, 70.0, 0.0])
    seed_deg = np.asarray(seed_deg, dtype=float)
    target = down_pose(x, y, z)
    q, err, tilt = _solve(target, seed_deg, 0.2, iters)
    for looser in (0.005, 0.0):          # let it lean, then stop caring about direction at all
        if err <= max_err_m:
            break
        q, err, tilt = _solve(target, seed_deg, looser, iters)
    lim = joint_limits_deg()
    ok = all(lim[n][0] - 0.5 <= q[i] <= lim[n][1] + 0.5 for i, n in enumerate(config.ARM_JOINTS))
    return q, err, tilt, ok
