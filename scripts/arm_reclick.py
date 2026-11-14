"""The camera got bumped: redo only the camera half of the calibration. The arm is not used.

The marks' positions in the arm's frame are still right as long as the arm's base has not
moved, so there is no need to touch them again. This shows the camera picture; click the
same marks in the same order as before, and the mapping is refitted and saved.

    python scripts/arm_reclick.py
"""
import json

import cv2

from earshot import config
from earshot.arm.table import Table, leave_one_out_mm
from earshot.camera import Camera

saved = json.loads(config.CALIB_FILE.read_text())
marks = saved["marks"]
arm_xy = [m["arm_xy"] for m in marks]
n = len(marks)

cam = Camera().start()
img = cam.fresh_frame()
cam.stop()
clicks: list[tuple[int, int]] = []


def redraw():
    view = img.copy()
    for i, (u, v) in enumerate(clicks, 1):
        cv2.circle(view, (u, v), 8, (0, 0, 255), 2)
        cv2.putText(view, str(i), (u + 10, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    msg = f"click mark {len(clicks) + 1} of {n}" if len(clicks) < n else "Enter = accept"
    cv2.putText(view, msg + "   (u = undo, q = quit)", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 0), 2)
    cv2.imshow("re-click the marks, same order as before", view)


def on_mouse(event, u, v, *_):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < n:
        clicks.append((u, v))
        redraw()


cv2.namedWindow("re-click the marks, same order as before")
cv2.setMouseCallback("re-click the marks, same order as before", on_mouse)
redraw()
while True:
    k = cv2.waitKey(30) & 0xFF
    if k == ord("u") and clicks:
        clicks.pop()
        redraw()
    elif k == ord("q"):
        raise SystemExit("Stopped. Nothing saved.")
    elif k in (13, 10) and len(clicks) == n:
        break
cv2.destroyAllWindows()

size = (img.shape[1], img.shape[0])
table = Table.fit(clicks, arm_xy, saved["table_z"], size)
print("Fit error at each mark (mm):", [round(e, 1) for e in table.errors_mm(clicks, arm_xy)])
loo = leave_one_out_mm(clicks, arm_xy, saved["table_z"], size)
if loo:
    print("Error predicting each mark from the others (mm):", [round(e, 1) for e in loo])
for m, c in zip(marks, clicks):
    m["pixel"] = list(c)
table.save(marks)
print(f"Saved {config.CALIB_FILE}.")
