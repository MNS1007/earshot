"""Kinematics for top-down grasps with the SO-101.

Wraps LeRobot's placo-based RobotKinematics. The arm has five joints, so it cannot reach an
arbitrary pose; for picking off a table we only ask for a position plus "gripper pointing
straight down, turned to face away from the base", which five joints can do.

Frames: the URDF base frame. x forward from the base, y to the arm's left, z up.
Angles in degrees, in LeRobot's joint order (see config.ARM_JOINTS).
"""
from __future__ import annotations

import json
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
def joint_limits_deg() -> dict[str, tuple[float, float]]:
    """How far each joint can really go, in degrees, in the URDF's convention.

    The tighter of two limits: the URDF's, and the range the motor was swept through at
    calibration (LeRobot refuses commands outside it). On this arm the calibrated wrist range
    stops at about 71 degrees, short of the URDF's 95, so the gripper cannot always point
    straight down.
    """
    out = {}
    for j in ET.parse(config.URDF_PATH).getroot().iter("joint"):
        lim = j.find("limit")
        if j.get("name") in config.ARM_JOINTS and lim is not None:
            out[j.get("name")] = (np.rad2deg(float(lim.get("lower"))), np.rad2deg(float(lim.get("upper"))))
    if config.ARM_CALIBRATION.exists():
        cal = json.loads(config.ARM_CALIBRATION.read_text())
        for name in config.ARM_JOINTS:
            lo = (cal[name]["range_min"] - 2047) * 360.0 / 4095 + 2.0      # 2 degrees of margin
            hi = (cal[name]["range_max"] - 2047) * 360.0 / 4095 - 2.0
            out[name] = (max(out[name][0], lo), min(out[name][1], hi))
    return out


@lru_cache(maxsize=1)
def _solver():
    from lerobot.model.kinematics import RobotKinematics

    k = RobotKinematics(config.URDF_PATH, target_frame_name=config.EE_FRAME, joint_names=config.ARM_JOINTS)
    for name, (lo, hi) in joint_limits_deg().items():
        k.robot.set_joint_limits(name, np.deg2rad(lo), np.deg2rad(hi))
    k.solver.enable_joint_limits(True)
    return k


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


def ik_down(x: float, y: float, z: float, seed_deg=None, iters: int = 120, max_err_m: float = 0.004):
    """Joint angles that put the fingertips at (x, y, z), pointing as nearly down as the arm can.

    Position comes first. The solver is run from several starting poses and with the "point
    down" request at three strengths; among the answers that land within `max_err_m`, the one
    leaning least wins (ties go to the one nearest `seed_deg`, so consecutive calls along a
    path do not flip the elbow).

    Returns (joints_deg, position_error_m, tilt_deg, within_limits).
    """
    pan = np.rad2deg(np.arctan2(y, x))
    lim = joint_limits_deg()
    seeds = [] if seed_deg is None else [np.asarray(seed_deg, dtype=float)]
    seeds += [np.array([pan, a, b, c, 0.0]) for a, b, c in ((-20, 40, 50), (20, 0, 55), (45, -40, 40), (-45, 75, 35))]
    target = down_pose(x, y, z)
    best = None
    for seed in seeds:
        for weight in (0.2, 0.02, 0.0):
            q, err, tilt = _solve(target, seed, weight, iters)
            ok = all(lim[n][0] - 0.5 <= q[i] <= lim[n][1] + 0.5 for i, n in enumerate(config.ARM_JOINTS))
            near = 0.0 if seed_deg is None else float(np.max(np.abs(q - seeds[0])))
            score = (not ok, err > max_err_m, round(max(err - max_err_m, 0.0), 4), round(tilt / 3.0), near)
            if best is None or score < best[0]:
                best = (score, q, err, tilt, ok)
            if ok and err <= max_err_m and tilt < 3.0:
                return q, err, tilt, ok
    _, q, err, tilt, ok = best
    return q, err, tilt, ok
