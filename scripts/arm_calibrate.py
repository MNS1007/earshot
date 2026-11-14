"""Camera-to-arm calibration by hand. The motors stay OFF for the whole script.

1. Put 5 or 6 small tape marks on the work area, spread out, not in one line.
2. For each mark, move the gripper tip onto it by hand and press Enter. The script reads the
   joint angles and works out where the tip is in the arm's own frame.
3. Move the arm out of the camera's view and press Enter. A window shows the camera picture:
   click the marks in the same order.
The fit, its accuracy and a marked-up picture are saved under data/.

    python scripts/arm_calibrate.py            # 6 marks
    python scripts/arm_calibrate.py --marks 5
"""
import argparse

import cv2
import numpy as np

from earshot import config
from earshot.arm import kin, robot as arm
from earshot.arm.table import Table, leave_one_out_mm
from earshot.arm import drift
from earshot.camera import Camera

ap = argparse.ArgumentParser()
ap.add_argument("--marks", type=int, default=6)
args = ap.parse_args()

print("Connecting with the motors off. Hold the arm: it will be limp.")
bot = arm.open_limp()
tips, joints = [], []
try:
    for i in range(1, args.marks + 1):
        input(f"\nMark {i} of {args.marks}: put the closed gripper tip on the mark, pointing down, then press Enter...")
        q = arm.read_joints(bot)
        x, y, z = kin.fk(q)[:3, 3]
        tips.append((x, y, z))
        joints.append([round(float(v), 2) for v in q])
        print(f"  joints {np.round(q, 1)}  ->  tip at x {x * 100:.1f} cm, y {y * 100:.1f} cm, z {z * 100:.1f} cm")
    input("\nNow move the arm out of the camera's view (fold it back) and press Enter...")
finally:
    arm.close(bot, limp=True)

zs = [t[2] for t in tips]
table_z = float(np.median(zs))
print(f"\nTable height in the arm's frame: {table_z * 100:.1f} cm (marks ranged {min(zs) * 100:.1f} to {max(zs) * 100:.1f}).")
if max(zs) - min(zs) > 0.015:
    print("  The heights differ by more than 1.5 cm. The model and the arm may not agree; tell Claude.")

cam = Camera().start()
img = cam.fresh_frame()
cam.stop()
clicks: list[tuple[int, int]] = []


def redraw():
    view = img.copy()
    for n, (u, v) in enumerate(clicks, 1):
        cv2.circle(view, (u, v), 8, (0, 0, 255), 2)
        cv2.putText(view, str(n), (u + 10, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    msg = f"click mark {len(clicks) + 1} of {args.marks}" if len(clicks) < args.marks else "Enter = accept, u = undo"
    cv2.putText(view, msg + "   (u = undo, q = quit)", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 0), 2)
    cv2.imshow("calibrate", view)


def on_mouse(event, u, v, *_):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < args.marks:
        clicks.append((u, v))
        redraw()


cv2.namedWindow("calibrate")
cv2.setMouseCallback("calibrate", on_mouse)
redraw()
while True:
    k = cv2.waitKey(30) & 0xFF
    if k == ord("u") and clicks:
        clicks.pop()
        redraw()
    elif k == ord("q"):
        raise SystemExit("Stopped. Nothing saved.")
    elif k in (13, 10) and len(clicks) == args.marks:
        break
cv2.destroyAllWindows()

arm_xy = [(t[0], t[1]) for t in tips]
size = (img.shape[1], img.shape[0])
table = Table.fit(clicks, arm_xy, table_z, size)
fit_err = table.errors_mm(clicks, arm_xy)
loo = leave_one_out_mm(clicks, arm_xy, table_z, size)
print("\nFit error at each mark (mm):", [round(e, 1) for e in fit_err])
if loo:
    print("Error predicting each mark from the others (mm):", [round(e, 1) for e in loo],
          f"  worst {max(loo):.1f}")
marks = [{"pixel": list(c), "arm_xy": list(a), "z": t[2], "joints": j} for c, a, t, j in zip(clicks, arm_xy, tips, joints)]
table.save(marks)
drift.save_reference(img)          # so a nudged camera can be corrected automatically later
out = config.DATA / "calibration.jpg"
view = img.copy()
for n, (u, v) in enumerate(clicks, 1):
    cv2.circle(view, (u, v), 8, (0, 0, 255), 2)
    cv2.putText(view, f"{n}: {arm_xy[n - 1][0] * 100:.0f},{arm_xy[n - 1][1] * 100:.0f} cm", (u + 10, v - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.imwrite(str(out), view)
print(f"\nSaved {config.CALIB_FILE} and {out}.")
