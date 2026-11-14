"""Camera picture to table position.

Everything the arm picks lies on the table, so a point in the camera picture maps to one
point on the table. That mapping (a homography) is fitted once from a few marks whose
position is known in both the picture and the arm's own frame, and saved to disk.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from earshot import config


class Table:
    """Pixel <-> arm-frame (x, y) on the table plane, plus the table's height."""

    def __init__(self, H: np.ndarray, table_z: float, image_size: tuple[int, int]):
        self.H, self.table_z, self.image_size = np.asarray(H, float), float(table_z), tuple(image_size)
        self._Hinv = np.linalg.inv(self.H)

    @staticmethod
    def fit(pixels, arm_xy, table_z: float, image_size) -> "Table":
        """Fit from matching points: `pixels` [[u, v], ...] and `arm_xy` [[x, y], ...] in metres."""
        px, xy = np.asarray(pixels, np.float64), np.asarray(arm_xy, np.float64)
        if len(px) < 4:
            raise ValueError("Calibration needs at least 4 marks.")
        H, _ = cv2.findHomography(px, xy, method=0)
        if H is None:
            raise ValueError("Could not fit the marks. Are three or more of them on one line?")
        return Table(H, table_z, image_size)

    def to_arm(self, u: float, v: float) -> tuple[float, float]:
        """Pixel -> (x, y) in metres in the arm's base frame."""
        p = self.H @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        p = self._Hinv @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def errors_mm(self, pixels, arm_xy) -> list[float]:
        """Distance in mm between each mark and where the fit puts it."""
        return [1000 * float(np.hypot(*(np.array(self.to_arm(*p)) - np.array(a)))) for p, a in zip(pixels, arm_xy)]

    def save(self, marks: list[dict] | None = None, path=config.CALIB_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"H": self.H.tolist(), "table_z": self.table_z,
                                    "image_size": list(self.image_size), "marks": marks or []}, indent=2))

    @staticmethod
    def load(path=config.CALIB_FILE) -> "Table":
        if not path.exists():
            raise RuntimeError(f"No camera-to-arm calibration at {path}. Run scripts/arm_calibrate.py first.")
        d = json.loads(path.read_text())
        return Table(np.array(d["H"]), d["table_z"], d["image_size"])


def leave_one_out_mm(pixels, arm_xy, table_z: float, image_size) -> list[float]:
    """Honest accuracy estimate: predict each mark from a fit that never saw it."""
    out = []
    for i in range(len(pixels)):
        rest_p = [p for j, p in enumerate(pixels) if j != i]
        rest_a = [a for j, a in enumerate(arm_xy) if j != i]
        if len(rest_p) < 4:
            return []
        t = Table.fit(rest_p, rest_a, table_z, image_size)
        out.append(t.errors_mm([pixels[i]], [arm_xy[i]])[0])
    return out
