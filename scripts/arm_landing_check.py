"""After a pick: draw where the fixed jaw was meant to land on the touch-down picture, with a 1 cm grid.

    python scripts/arm_landing_check.py U V        # U V = the "near edge on the table" pixel the pick printed

Green cross: where the fixed jaw's tip was aimed. Blue cross: the item's near edge. Orange
lines cross at the aim; grid lines are 1 cm apart on the table. Compare with where the tip is.
"""
import sys

import cv2
import numpy as np

from earshot import config
from earshot.arm import motion
from earshot.arm.table import Table

u, v = float(sys.argv[1]), float(sys.argv[2])
t = Table.load()
c = np.array(t.to_arm(u, v))
away = np.array(t.to_arm(u, v - 12.0)) - c
away /= np.linalg.norm(away)
aim = c - away * motion.RAKE_LEAD
shots = sorted((config.DATA / "pick_frames").glob("3_down*.jpg"))
im = cv2.imread(str(shots[-1]))
for d in range(-4, 5):
    colour = (0, 120, 255) if d == 0 else (200, 200, 0)
    for a, b in (((aim[0] + d * 0.01, aim[1] - 0.04), (aim[0] + d * 0.01, aim[1] + 0.04)),
                 ((aim[0] - 0.04, aim[1] + d * 0.01), (aim[0] + 0.04, aim[1] + d * 0.01))):
        cv2.line(im, tuple(int(k) for k in t.to_pixel(*a)), tuple(int(k) for k in t.to_pixel(*b)), colour, 1)
ap = t.to_pixel(*aim)
cv2.drawMarker(im, (int(ap[0]), int(ap[1])), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
cv2.drawMarker(im, (int(u), int(v)), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 12, 1)
x0, y0 = max(0, int(ap[0]) - 150), max(0, int(ap[1]) - 130)
out = config.DATA / "landing_check.jpg"
cv2.imwrite(str(out), cv2.resize(im[y0:y0 + 260, x0:x0 + 300], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
print(out)
