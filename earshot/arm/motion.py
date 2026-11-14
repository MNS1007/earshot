"""Arm motion: slow joint moves, and the one pick-and-place the whole system uses.

Every move is cut into small steps sent at a fixed rate, and LeRobot's own per-command limit
(`max_relative_target`) is left on underneath, so a bad target cannot make the arm lunge.
Positions are given as camera pixels and converted with the saved calibration (table.py).
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from earshot import config
from earshot.arm import kin, robot as arm
from earshot.arm.table import Table

RATE_HZ = 30
GRIP_OPEN = 45.0          # 0-100, LeRobot's gripper scale: used for letting go
ABOVE_TABLE_APPROACH = 0.025  # lowered to here normally; from here down the arm feels for the table
ABOVE_TABLE_GRASP = 0.006     # height for setting an item back down on the table
ABOVE_TABLE_HOVER = 0.06
ABOVE_TABLE_DROP = 0.14    # over a tall container; the gripper leans at this height
ABOVE_TABLE_LOW_DROP = 0.08  # over a bowl or a box
RAKE_LEAD = 0.02             # the fixed jaw lands this far in front of the item's near side (twice the pointing error)
# Measured on 2026-11-14 from a touch-down picture with a 1 cm grid drawn on the table map: the
# fixed jaw's tip came down 0.45 cm further along the jaw line and 1.7 cm to its left of the aim.
TIP_ALONG = 0.0045
TIP_LEFT = 0.017
CLEARANCE = 0.012            # spare room each side of the item between the open fingertips
GRIP_DEPTH = 0.016           # how far behind the item's near edge the jaws close (half the depth of a small item)
GRIP_DEPTH_MAX = 0.03        # deepest the jaws close behind the near edge
NARROW_FACE = 0.045          # an item narrower than this across the view is taken to be end-on to the camera
MAX_GAP = 0.095              # widest fingertip gap at which the swinging jaw still closes near the table
EMPTY_GRIPPER_READS = 2.0    # what the gripper position reads when it has closed on nothing (seen on every miss today)
RAKE_STEP_BACK = 0.015       # how far to step back when the fixed jaw comes down on top of the item
ON_TABLE = 0.014             # a fingertip measured lower than this is on the table, not on an item
RAKE_GAP = 0.085             # fingertip gap while coming down: about the widest at which the swinging jaw still reaches the table
MAX_PLACE_REACH = 0.30       # metres from the base; a destination further out is approached as far as this


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
        self._at = (0.0, 0.0, ABOVE_TABLE_HOVER)
        self.last_lean = 0.0
        self.jaw_deg: Optional[float] = None    # direction the jaws close along; None = toward the base
        self.abort = threading.Event()          # set from another thread to stop the current move
        self._closed = 5.0                      # the gripper setting last used to close on an item
        self.on_stage = None                    # optional callback(name) at each stage of a pick, for recording

    # -- low level -------------------------------------------------------------------------
    def _send(self, q: np.ndarray, grip: float) -> None:
        if self.abort.is_set():
            raise Stopped("stopped")
        action = {f"{j}.pos": float(v) for j, v in zip(config.ARM_JOINTS, arm.to_motor(self.robot, q))}
        action[f"{arm.GRIPPER}.pos"] = float(grip)
        self.robot.send_action(action)

    def _glide(self, target: np.ndarray, timeout: float) -> None:
        """Like move_joints but gives up waiting after `timeout`, for moves that may meet the table."""
        start = arm.read_joints(self.robot)
        n = max(2, int(np.ceil(np.max(np.abs(target - start)) / (0.3 + 3.0 * self.speed))))
        for i in range(1, n + 1):
            self._send(start + (target - start) * i / n, self._grip_now)
            time.sleep(1.0 / RATE_HZ)
        self._settle(target, self._grip_now, tol_deg=0.6, timeout=timeout)

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
        self._settle(target, grip)

    def _settle(self, target: np.ndarray, grip: float, tol_deg: float = 1.5, timeout: float = 4.0) -> None:
        """Keep asking for `target` until the arm is really there (commands are cut short when it lags)."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if np.max(np.abs(arm.read_joints(self.robot) - target)) < tol_deg:
                return
            self._send(target, grip)
            time.sleep(1.0 / RATE_HZ)

    def set_gripper(self, value: float, wait: float = 0.3) -> None:
        """Open or close to `value`, stopping early when the fingers stall on an item."""
        q = arm.read_joints(self.robot)
        last, still, t0 = arm.read_gripper(self.robot), 0, time.time()
        while time.time() - t0 < 2.5:
            self._send(q, value)
            time.sleep(1.0 / RATE_HZ)
            now = arm.read_gripper(self.robot)
            if abs(now - value) < 2.0:
                break
            still = still + 1 if abs(now - last) < 0.4 else 0
            if still >= 8:                      # about a quarter of a second without moving: it is gripping
                break
            last = now
        self._grip_now = value
        time.sleep(wait)

    def _solve(self, x: float, y: float, above_table: float, seed=None) -> np.ndarray:
        z = self.table.table_z + above_table
        q, err, lean, ok = kin.ik_down(x, y, z, seed_deg=seed, jaw_deg=self.jaw_deg)
        if not ok or err > 0.008:
            raise ReachError(f"Cannot reach x {x * 100:.0f} cm, y {y * 100:.0f} cm at {above_table * 100:.0f} cm "
                             f"above the table (off by {err * 1000:.0f} mm).")
        self.last_lean = lean
        return q

    def move_xyz(self, x: float, y: float, above_table: float) -> float:
        """Fingertips to (x, y) at a height above the table. Returns the gripper's lean in degrees.

        Close to the base the arm cannot get as high, so if the height is out of reach it
        settles for the highest it can manage there, down to 3 cm.
        """
        for h in (above_table, 0.045, 0.03):
            if h > above_table:
                continue
            try:
                q = self._solve(x, y, h)
            except ReachError:
                if h == 0.03 or above_table <= 0.03:
                    raise
                continue
            self.move_joints(q)
            self._at = (x, y, h)
            return self.last_lean

    def move_height(self, above_table: float) -> float:
        """Straight up or down over the current spot, in 1 cm slices so the tip does not swing sideways.

        If the arm cannot reach the full height here it stops at the last slice it could.
        """
        x, y, h0 = self._at
        q = arm.read_joints(self.robot)
        reached = h0
        for h in np.linspace(h0, above_table, max(2, int(abs(above_table - h0) / 0.01) + 1))[1:]:
            try:
                q = self._solve(x, y, float(h), seed=q)
            except ReachError:
                break
            self.move_joints(q)
            reached = float(h)
        self._at = (x, y, reached)
        return self.last_lean

    def tip_height(self) -> float:
        """Where the fingertip really is above the table, from the joints as measured (not as commanded)."""
        return float(kin.fk(arm.read_joints(self.robot))[2, 3] - self.table.table_z)

    def follow_camera(self, frame) -> str:
        """Re-find the calibration marks in `frame` and correct for a nudged camera. Returns a short report."""
        from earshot.arm import drift

        table, used, shift = drift.refit(frame)
        if used >= 2:
            self.table = table
            return f"camera checked: {used} marks agree it moved {shift:.0f} px since calibration"
        return "camera check skipped: fewer than two marks visible, using the saved calibration"

    def go_home(self) -> None:
        self.move_joints(self.home)

    # -- the one call the rest of the system uses --------------------------------------------
    def jaw_across(self, pick_px, end_a_px, end_b_px) -> float:
        """Jaw direction that closes across an item's narrow side, given the two ends of its long side."""
        ax, ay = self.table.to_arm(*end_a_px)
        bx, by = self.table.to_arm(*end_b_px)
        px, py = self.table.to_arm(*pick_px)
        across = np.rad2deg(np.arctan2(by - ay, bx - ax)) + 90.0     # square to the long side
        radial = np.rad2deg(np.arctan2(py, px))
        turn = (across - radial + 90.0) % 180.0 - 90.0               # jaws are symmetric: never turn more than 90
        return float(radial + turn)

    def footprint(self, outline_px) -> Optional[tuple[float, float, Optional[float], float]]:
        """Middle of an item's base and the direction of its long side, from its ground-contact outline.

        The outline pixels are true table points, so they are mapped to the table exactly,
        whatever the camera's tilt or the item's height. Seen from above at an angle, the
        visible part of a box's base is two edges meeting at the nearest corner, and the
        outline's two ends are opposite corners of the base: its middle is half way between
        them. (For a round item the ends are the two sides of the circle: same answer.)
        When only one edge shows (the item is square-on to the camera) the middle of the base
        cannot be known from the outline, and None is returned.
        Returns (x, y, long_side_direction_deg or None, short_side_m) in the arm's frame.
        """
        pts = np.array([self.table.to_arm(u, v) for u, v in outline_px])
        if len(pts) < 12:
            return None
        k = max(2, len(pts) // 20)
        a, b = pts[:k].mean(axis=0), pts[-k:].mean(axis=0)         # the two ends of the outline
        chord = b - a
        length = float(np.linalg.norm(chord))
        if length < 0.01:
            return None
        normal = np.array([-chord[1], chord[0]]) / length
        off = (pts - a) @ normal
        i = int(np.argmax(np.abs(off)))
        corner = pts[i]
        if abs(off[i]) < 0.2 * length:                             # nearly straight: one edge only
            return None
        e1, e2 = corner - a, b - corner
        long_edge = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
        ratio = max(np.linalg.norm(e1), np.linalg.norm(e2)) / max(1e-6, min(np.linalg.norm(e1), np.linalg.norm(e2)))
        direction = float(np.rad2deg(np.arctan2(long_edge[1], long_edge[0]))) if ratio > 1.25 else None
        centre = (a + b) / 2.0
        return float(centre[0]), float(centre[1]), direction, float(min(np.linalg.norm(e1), np.linalg.norm(e2)))

    def lower_until_touch(self, floor: float = -0.012, slice_m: float = 0.004) -> float:
        """Lower over the current spot until the fingertips press on the table. Returns the height reached.

        No height is assumed: the arm goes down in thin slices and after each one compares
        where its joints are with where they were told to be. In free air that gap stays
        small and steady; when the fingers meet the table it jumps, and the arm stops there.
        `floor` (below the calibrated table height) is only a backstop.
        """
        x, y, h = self._at
        q = arm.read_joints(self.robot)
        gaps: list[float] = []
        while h - slice_m >= floor:
            h -= slice_m
            try:
                q = self._solve(x, y, h, seed=q)
            except ReachError:
                break
            self._glide(q, timeout=0.45)
            gap = float(np.max(np.abs(arm.read_joints(self.robot)[:4] - q[:4])))
            if len(gaps) >= 3 and gap > float(np.median(gaps)) + 1.2:
                break                                              # pressing on the table
            gaps.append(gap)
        self._at = (x, y, h)
        return h

    def _sideways_grasp(self, outline_px, away: np.ndarray, left: np.ndarray):
        """Plan a centre grasp across the camera's view from an item's ground-contact outline.

        The outline's two ends are the item's leftmost and rightmost points on the table as the
        camera sees it, so the item's middle (sideways) and its width are known exactly. Depth
        is taken a little way in from the near edge, which is inside any item deeper than about
        2.5 cm. Returns (middle xy, fingertip gap, unit vector the gap opens along), or None when
        there is no outline, the item is too wide for the gripper, or the wrist cannot turn there.
        """
        if outline_px is None or len(outline_px) < 12:
            return None
        pts = np.array([self.table.to_arm(u, v) for u, v in outline_px])
        across, depth = pts @ left, pts @ away
        width = float(across.max() - across.min())
        gap = width + 2 * CLEARANCE
        if width < 0.008 or gap > MAX_GAP:
            return None
        mid_across = float((across.max() + across.min()) / 2.0)
        near = float(np.percentile(depth[np.abs(across - mid_across) < max(0.004, 0.2 * width)], 10))
        # How far in from the near edge to close. The far side cannot be seen, so this is a
        # judgement: an item showing its narrow face runs away from the camera at least as far
        # as it is wide, so grip one width in; a wide item is gripped just behind its front.
        depth_in = min(width, GRIP_DEPTH_MAX) if width < NARROW_FACE else GRIP_DEPTH
        middle = left * mid_across + away * (near + depth_in)
        best = None
        for vec in (left, -left):                 # the gap can open to either side: take the easier wrist turn
            fixed = middle - vec * gap / 2.0
            q, err, _, ok = kin.ik_down(fixed[0], fixed[1], self.table.table_z + ABOVE_TABLE_APPROACH,
                                        jaw_deg=float(np.rad2deg(np.arctan2(vec[1], vec[0]))))
            if ok and err < 0.008 and (best is None or abs(q[4]) < best[0]):
                best = (abs(q[4]), vec)
        return None if best is None else (middle, gap, best[1])

    def pick_place(self, pick_px, place_px, drop: bool, tall: bool = False, long_side_px=None, outline_px=None) -> bool:
        """Pick the item at `pick_px` and put it at `place_px` (camera pixels).

        `pick_px` should be where the item's near side meets the table, which is what the
        pointing model gives. drop=True releases from above (into a container; `tall` for a
        cup); drop=False sets it down on the table. `long_side_px` and `outline_px` are accepted
        for callers that measure them but are not needed by the rake grasp used here.
        Returns False, without going to the destination, when the gripper closed on nothing,
        and True when the motion finished. Whether the item actually arrived is for the camera
        to check afterwards.
        """
        px, py = self.table.to_arm(*pick_px)
        qx, qy = self.table.to_arm(*place_px)
        far = float(np.hypot(qx, qy))
        if far > MAX_PLACE_REACH:                # e.g. a container whose near edge is in reach but whose middle is not
            qx, qy = qx * MAX_PLACE_REACH / far, qy * MAX_PLACE_REACH / far

        away = np.array(self.table.to_arm(pick_px[0], pick_px[1] - 12.0)) - np.array([px, py])
        away = away / np.linalg.norm(away)        # along the table, from the camera toward the item
        left = np.array([-away[1], away[0]])      # along the table, across the camera's view
        side = self._sideways_grasp(outline_px, away, left)
        if side is not None:
            # Centre grasp, jaws closing across the camera's view. Both sides of the item are in
            # plain sight where it meets the table, so its middle and its width are measured, not
            # estimated: the gripper opens a little wider than the item, comes down with the item
            # in the middle of the gap, goes to the table and closes there.
            middle, gap, line_vec = side
            line = float(np.rad2deg(np.arctan2(line_vec[1], line_vec[0])))
            fixed = middle - line_vec * gap / 2.0
        else:
            # Rake grasp, for items whose outline could not be traced: the fixed jaw comes down
            # just in front of the item's near side and the other jaw sweeps the item up to it.
            line_vec, gap = away, RAKE_GAP
            line = float(np.rad2deg(np.arctan2(away[1], away[0])))
            _, err, _, ok = kin.ik_down(px, py, self.table.table_z + ABOVE_TABLE_APPROACH, jaw_deg=line)
            if not ok or err > 0.008:             # the wrist cannot turn that far here: rake outward from the base
                line = float(np.rad2deg(np.arctan2(py, px)))
                line_vec = np.array([np.cos(np.deg2rad(line)), np.sin(np.deg2rad(line))])
            fixed = np.array([px, py]) - line_vec * RAKE_LEAD
        self.jaw_deg = line
        # Where the fixed jaw's tip really lands differs from where the model puts it by a small
        # fixed amount in the gripper's own frame (measured, see TIP_ALONG and TIP_LEFT).
        across = np.array([-line_vec[1], line_vec[0]])
        px, py = fixed - line_vec * TIP_ALONG - across * TIP_LEFT
        away = line_vec if side is None else away  # direction to step back if the jaw lands on the item
        closed = 3.0 * (1.0 - self.grip)
        self._closed = closed
        self.set_gripper(kin.gripper_for_gap(gap), wait=0.2)
        self.move_xyz(px, py, ABOVE_TABLE_HOVER)
        self._stage("1 above")
        for attempt in range(3):
            self.move_height(ABOVE_TABLE_APPROACH)
            self.lower_until_touch()
            height = self.tip_height()
            self._stage(f"3 down try {attempt + 1}, tip measured {height * 1000:.0f} mm above the table, "
                        f"lean {self.last_lean:.0f} deg")
            if height < ON_TABLE:
                break
            # The fingertip stopped well above the table: it is resting on the item, not beside
            # it. Lift, step back toward the camera, and come down again.
            self.move_height(ABOVE_TABLE_HOVER)
            if side is not None:
                break                             # centred on a measured outline: stepping back would not help
            px, py = np.array([px, py]) - away * RAKE_STEP_BACK
            self.move_xyz(px, py, ABOVE_TABLE_HOVER)
        self.set_gripper(closed)
        self._stage(f"4 closed, gripper reads {arm.read_gripper(self.robot):.0f}")
        self.jaw_deg = None                      # lifting does not need the wrist held at an angle
        self.move_height(ABOVE_TABLE_HOVER)
        self._stage(f"5 lifted, gripper reads {arm.read_gripper(self.robot):.0f}")
        if not self.holding_something():         # the fingers met: nothing is between them
            self.jaw_deg = None
            self.set_gripper(GRIP_OPEN, wait=0.2)
            self.go_home()
            return False
        self.jaw_deg = None                      # carry and release with the wrist relaxed
        if drop:
            self.move_xyz(qx, qy, ABOVE_TABLE_DROP if tall else ABOVE_TABLE_LOW_DROP)
            self._stage("6 over the container")
        else:
            self.move_xyz(qx, qy, ABOVE_TABLE_HOVER)
            self.move_height(ABOVE_TABLE_GRASP)
        self.set_gripper(GRIP_OPEN, wait=0.3)
        self._stage("7 released")
        if not drop:
            self.move_height(ABOVE_TABLE_HOVER)
        self.go_home()
        return True

    def _stage(self, name: str) -> None:
        if self.on_stage:
            self.on_stage(name)

    def holding_something(self) -> bool:
        """After closing: a gripper that stopped short of its target is holding an item."""
        # Shut on nothing, this gripper has read 2 every time. Holding a block by a thin wall it
        # read 3, so the margin is small on purpose: letting go of something it has is worse
        # than carrying nothing, which the camera check catches afterwards anyway.
        return arm.read_gripper(self.robot) > EMPTY_GRIPPER_READS + 0.6

    def back_off(self) -> None:
        """After a stop: let go of anything held, lift clear, and return to rest."""
        self.abort.clear()
        self.jaw_deg = None
        self.set_gripper(GRIP_OPEN, wait=0.1)
        x, y, _ = self._at
        if x or y:
            try:
                self.move_xyz(x, y, ABOVE_TABLE_HOVER)
            except ReachError:
                pass
        self.go_home()

    def close(self) -> None:
        """Return to rest and switch the motors off. Support the arm if it is not resting."""
        self.abort.clear()
        try:
            self.go_home()
            self.set_gripper(self.home_grip, wait=0.2)
        finally:
            arm.close(self.robot, limp=True)


class Stopped(RuntimeError):
    """The move was stopped from outside (pause, retry, or shutting down)."""


class ReachError(RuntimeError):
    """The requested point is outside what the arm can reach."""
