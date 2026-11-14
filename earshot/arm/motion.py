"""Arm motion: slow joint moves, and the one pick-and-place the whole system uses.

Every move is cut into small steps sent at a fixed rate, and LeRobot's own per-command limit
(`max_relative_target`) is left on underneath, so a bad target cannot make the arm lunge.
Positions are given as camera pixels and converted with the saved calibration (table.py).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from earshot import config
from earshot.arm import kin, robot as arm
from earshot.arm.table import Table

RATE_HZ = 30
GRIP_OPEN = 70.0          # 0-100, LeRobot's gripper scale
ABOVE_TABLE_GRASP = 0.012  # fingertip height above the table when closing, metres
ABOVE_TABLE_HOVER = 0.06
ABOVE_TABLE_DROP = 0.14    # over a tall container; the gripper leans at this height
ABOVE_TABLE_LOW_DROP = 0.08  # over a bowl or a box


class Arm:
    """A connected, holding arm plus its calibration. Create once, reuse for every pick."""

    def __init__(self, speed: float = 0.5, table: Optional[Table] = None):
        self.table = table or Table.load()
        self.robot = arm.open_holding(max_step_deg=6.0)
        self.speed = speed                      # 0.2 slow ... 1.0 fast
        self.grip = 0.6                         # 0 gentle ... 1 firm
        self.home = arm.read_joints(self.robot)  # where the arm was resting; it returns here
        self.home_grip = arm.read_gripper(self.robot)
        self._grip_now = self.home_grip

    # -- low level -------------------------------------------------------------------------
    def _send(self, q: np.ndarray, grip: float) -> None:
        action = {f"{j}.pos": float(v) for j, v in zip(config.ARM_JOINTS, arm.to_motor(self.robot, q))}
        action[f"{arm.GRIPPER}.pos"] = float(grip)
        self.robot.send_action(action)

    def move_joints(self, target: np.ndarray, grip: Optional[float] = None) -> None:
        """Glide to `target` (degrees). Blocks until there."""
        grip = self._grip_now if grip is None else grip
        start = arm.read_joints(self.robot)
        step = 0.3 + 3.0 * self.speed                       # degrees per tick, largest joint (0.2 -> about 27 deg/s)
        n = max(2, int(np.ceil(np.max(np.abs(target - start)) / step)))
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / n)           # ease in and out
            self._send(start + s * (target - start), grip)
            time.sleep(1.0 / RATE_HZ)
        self._settle(target)

    def _settle(self, target: np.ndarray, tol_deg: float = 2.0, timeout: float = 1.5) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if np.max(np.abs(arm.read_joints(self.robot) - target)) < tol_deg:
                return
            time.sleep(0.03)

    def set_gripper(self, value: float, wait: float = 0.6) -> None:
        q = arm.read_joints(self.robot)
        n = 10
        for i in range(1, n + 1):
            self._send(q, self._grip_now + (value - self._grip_now) * i / n)
            time.sleep(1.0 / RATE_HZ)
        self._grip_now = value
        time.sleep(wait)

    def move_xyz(self, x: float, y: float, above_table: float) -> tuple[float, float]:
        """Fingertips to (x, y) at a height above the table. Returns (position error m, lean deg)."""
        q, err, lean, ok = kin.ik_down(x, y, self.table.table_z + above_table, seed_deg=arm.read_joints(self.robot))
        if not ok or err > 0.01:
            q, err, lean, ok = kin.ik_down(x, y, self.table.table_z + above_table)   # retry from the standard guess
        if not ok or err > 0.01:
            raise ReachError(f"Cannot reach x {x * 100:.0f} cm, y {y * 100:.0f} cm at {above_table * 100:.0f} cm "
                             f"above the table (off by {err * 1000:.0f} mm).")
        self.move_joints(q)
        return err, lean

    def go_home(self) -> None:
        self.move_joints(self.home)

    # -- the one call the rest of the system uses --------------------------------------------
    def pick_place(self, pick_px, place_px, drop: bool, tall: bool = False) -> bool:
        """Pick the item at `pick_px` and put it at `place_px` (camera pixels).

        drop=True releases from above (into a container; `tall` for a cup); drop=False sets it
        down on the table. Returns True when the motion finished. Whether the item actually
        arrived is for the camera to check afterwards.
        """
        px, py = self.table.to_arm(*pick_px)
        qx, qy = self.table.to_arm(*place_px)
        closed = max(0.0, 22.0 - 22.0 * self.grip)
        self.set_gripper(GRIP_OPEN, wait=0.2)
        self.move_xyz(px, py, ABOVE_TABLE_HOVER)
        self.move_xyz(px, py, ABOVE_TABLE_GRASP)
        self.set_gripper(closed)
        self.move_xyz(px, py, ABOVE_TABLE_HOVER)
        if drop:
            self.move_xyz(qx, qy, ABOVE_TABLE_DROP if tall else ABOVE_TABLE_LOW_DROP)
        else:
            self.move_xyz(qx, qy, ABOVE_TABLE_HOVER)
            self.move_xyz(qx, qy, ABOVE_TABLE_GRASP + 0.004)
        self.set_gripper(GRIP_OPEN, wait=0.3)
        if not drop:
            self.move_xyz(qx, qy, ABOVE_TABLE_HOVER)
        self.go_home()
        return True

    def holding_something(self) -> bool:
        """After closing: a gripper that stopped short of its target is holding an item."""
        return arm.read_gripper(self.robot) > max(0.0, 22.0 - 22.0 * self.grip) + 4.0

    def close(self) -> None:
        """Return to rest and switch the motors off. Support the arm if it is not resting."""
        try:
            self.go_home()
            self.set_gripper(self.home_grip, wait=0.2)
        finally:
            arm.close(self.robot, limp=True)


class ReachError(RuntimeError):
    """The requested point is outside what the arm can reach."""
