"""Offline check, no arm needed: where on the table can the arm do a top-down grasp?

Solves a grid of table points at grasp and hover height and prints, for each row of the grid,
how far the gripper has to lean from straight down to get there, or that it cannot reach.
Use it to decide where the work area goes before any motor moves.
"""
import numpy as np

from earshot import config
from earshot.arm import kin

XS = np.arange(0.10, 0.36, 0.03)     # metres forward of the base
YS = np.arange(-0.18, 0.181, 0.06)   # metres left/right


def cell(x, y, z):
    q, err, tilt, ok = kin.ik_down(x, y, z)
    if not ok or err >= 0.004:
        return " -- ", err, tilt, q
    return f"{tilt:3.0f} ", err, tilt, q


for z, label in ((config.GRASP_Z, "grasp height"), (config.HOVER_Z, "hover height"), (config.DROP_Z, "drop height over a cup")):
    print(f"\n{label}  z = {z * 100:.1f} cm   (number = gripper lean in degrees from straight down; -- = cannot reach within 4 mm)")
    print("  x\\y   " + "".join(f"{y * 100:+5.0f}" for y in YS))
    worst = 0.0
    for x in XS:
        row = [cell(x, y, z) for y in YS]
        worst = max([worst] + [r[1] for r in row if r[0] != " -- "])
        print(f"  {x * 100:4.0f}   " + "".join(f" {r[0]}" for r in row))
    print(f"  worst position error among reachable cells: {worst * 1000:.1f} mm")

q, err, tilt, ok = kin.ik_down(0.22, 0.0, config.GRASP_Z)
print("\nexample, 22 cm straight ahead at grasp height:")
print("  joints (deg):", dict(zip(config.ARM_JOINTS, np.round(q, 1))))
print(f"  error {err * 1000:.1f} mm, tilt {tilt:.1f} deg, inside limits: {ok}")
print("  joint limits (deg):", {k: (round(a), round(b)) for k, (a, b) in kin.joint_limits_deg().items()})
