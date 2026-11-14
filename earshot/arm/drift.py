"""Keep the camera calibration right when the camera gets nudged.

At calibration time a small picture of each mark is saved. Later, each mark is searched for
again near where it used to be; the marks that are found clearly give their new pixel
positions, and the pixel-to-table mapping is refitted from them. Marks covered by an item
or a hand simply score badly and are left out. This handles shifts of a few centimetres on
the table, not a camera that has been carried somewhere else.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from earshot import config
from earshot.arm.table import Table

REF_FRAME = config.DATA / "calibration_frame.jpg"
PATCH = 22          # half-size of the saved picture of a mark, pixels
SEARCH = 90         # how far from its old place a mark is looked for, pixels
GOOD = 0.6          # match score needed to trust a mark (1.0 = identical)


def save_reference(frame: np.ndarray) -> None:
    """Call with the exact frame the marks were clicked on."""
    cv2.imwrite(str(REF_FRAME), frame, [cv2.IMWRITE_JPEG_QUALITY, 97])


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)


def locate_marks(frame: np.ndarray) -> list[dict]:
    """Where each calibration mark is now: [{"pixel": [u, v], "score": s, "arm_xy": [x, y]}, ...]."""
    saved = json.loads(config.CALIB_FILE.read_text())
    ref, now = _gray(cv2.imread(str(REF_FRAME))), _gray(frame)
    h, w = now.shape
    out = []
    for m in saved["marks"]:
        u, v = int(round(m["pixel"][0])), int(round(m["pixel"][1]))
        patch = ref[max(0, v - PATCH): v + PATCH + 1, max(0, u - PATCH): u + PATCH + 1]
        x0, y0 = max(0, u - SEARCH), max(0, v - SEARCH)
        window = now[y0: min(h, v + SEARCH + 1), x0: min(w, u + SEARCH + 1)]
        if patch.shape[0] < 2 * PATCH or patch.shape[1] < 2 * PATCH or window.shape[0] <= patch.shape[0]:
            out.append({"pixel": [u, v], "score": 0.0, "arm_xy": m["arm_xy"]})
            continue
        res = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        out.append({"pixel": [x0 + loc[0] + PATCH, y0 + loc[1] + PATCH], "score": float(score), "arm_xy": m["arm_xy"]})
    return out


def refit(frame: np.ndarray) -> tuple[Table, int, float]:
    """A Table for the camera as it is now. Returns (table, marks used, median shift in pixels).

    Falls back to the saved calibration when fewer than five marks can be found clearly.
    """
    saved = json.loads(config.CALIB_FILE.read_text())
    base = Table.load()
    if not REF_FRAME.exists():
        return base, 0, 0.0
    found = [(m, old) for m, old in zip(locate_marks(frame), saved["marks"]) if m["score"] >= GOOD]
    if len(found) < 5:
        return base, len(found), 0.0
    shift = float(np.median([np.hypot(m["pixel"][0] - o["pixel"][0], m["pixel"][1] - o["pixel"][1]) for m, o in found]))
    table = Table.fit([m["pixel"] for m, _ in found], [m["arm_xy"] for m, _ in found], saved["table_z"], saved["image_size"])
    return table, len(found), shift
