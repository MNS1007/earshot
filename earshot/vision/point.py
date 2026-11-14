# Find things in a photo of the table: point at named items, list loose items, check a placement, draw points.
"""Pointing: turn item descriptions into pixel coordinates in a table photo.

Every function takes an image either as a BGR numpy array (a camera frame) or as a path
to an image file. Points come back in pixels of that image, at the spot where the object
touches the surface it rests on, because the arm code projects them onto the table plane.
"""
from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from earshot import config
from earshot.vision import gemini

SEND_LONG_SIDE = 1536      # photos are shrunk to this before sending; points are normalised so nothing is lost

THINKING = "low"           # pointing with low thinking takes about 3 s instead of 10 and lands as well

BASE_POINT = (
    "A point is [y, x], each normalised to 0-1000 over the image (y down from the top, x from the left). "
    "Put the point where the object touches the surface it rests on (the middle of its base or of the "
    "part lying on the surface), not on its top."
)

POINTS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "index": {"type": "integer", "description": "number of the description this answers"},
            "point": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        },
        "required": ["index", "point"],
    },
}

ITEMS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "point": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        },
        "required": ["label", "point"],
    },
}

PLACED_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}, "ok": {"type": "boolean"}},
    "required": ["reason", "ok"],
}


def load(image: np.ndarray | str | Path) -> np.ndarray:
    """Return the image as a BGR array, reading it from disk if a path was given."""
    if isinstance(image, np.ndarray):
        return image
    loaded = cv2.imread(str(image))
    if loaded is None:
        raise FileNotFoundError(f"cannot read image {image}")
    return loaded


def _image_part(image_bgr: np.ndarray) -> dict:
    """Encode an image as an inline JPEG part, shrunk if it is very large."""
    height, width = image_bgr.shape[:2]
    scale = SEND_LONG_SIDE / max(height, width)
    if scale < 1:
        image_bgr = cv2.resize(image_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("could not encode the image as JPEG")
    return {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpeg.tobytes()).decode()}}


def _to_pixels(point, image_bgr: np.ndarray) -> tuple[float, float] | None:
    """Convert the model's [y, x] 0-1000 point to (x, y) pixels, or None if it is unusable."""
    if not isinstance(point, list) or len(point) != 2:
        return None
    y, x = (min(max(float(v), 0.0), 1000.0) for v in point)
    height, width = image_bgr.shape[:2]
    return round(x / 1000 * width, 1), round(y / 1000 * height, 1)


def point_at(image: np.ndarray | str | Path, descriptions: list[str],
             model: str = config.POINT_MODEL) -> list[dict]:
    """Point at each described object.

    Returns `{"label", "x", "y"}` per description found, in the order asked, with x and y
    in pixels of the given image at the object's base. Descriptions the model cannot find
    are left out. A description listed twice gets two different objects.
    """
    if not descriptions:
        return []
    image_bgr = load(image)
    listing = "\n".join(f"{i}: {d}" for i, d in enumerate(descriptions, 1))
    prompt = (
        "Point at each of these objects in the image:\n" + listing + "\n\n"
        "Answer with one entry per description you can see, giving its number as `index` and a `point`. "
        "Leave out any description that has no matching object in the image; never guess. "
        "If the same description appears more than once, give each one a different object. " + BASE_POINT
    )
    answer = gemini.generate(model, [_image_part(image_bgr), {"text": prompt}], POINTS_SCHEMA, temperature=0.0, thinking=THINKING)
    found: dict[int, dict] = {}
    for entry in answer:
        index, pixels = entry.get("index"), _to_pixels(entry.get("point"), image_bgr)
        if isinstance(index, int) and 1 <= index <= len(descriptions) and pixels and index not in found:
            found[index] = {"label": descriptions[index - 1], "x": pixels[0], "y": pixels[1]}
    return [found[i] for i in sorted(found)]


AXIS_SCHEMA = {
    "type": "object",
    "properties": {
        "elongated": {"type": "boolean", "description": "false if the object is about as wide as it is long"},
        "centre": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        "end_a": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
        "end_b": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
    },
    "required": ["elongated", "centre", "end_a", "end_b"],
}


def grasp_of(image: np.ndarray | str | Path, description: str, model: str = config.POINT_MODEL) -> dict | None:
    """Where to grip one object and which way it lies.

    Returns {"x", "y", "long_side": ((ax, ay), (bx, by)) or None} in pixels, or None when the
    object is not in the picture. `long_side` is the two ends of the object's longer side where
    it rests on the surface; None for objects with no clear longer side.
    """
    image_bgr = load(image)
    prompt = (
        f"Find this object in the image: {description}.\n"
        "Give `centre`: the middle of the part of it that rests on the surface. "
        "Give `end_a` and `end_b`: the two ends of its longer side, on the surface, measured along the object "
        "itself and not along the image. Set `elongated` to false if it is about as wide as it is long. "
        "If the object is not in the image, answer with all points [0, 0]. " + BASE_POINT
    )
    a = gemini.generate(model, [_image_part(image_bgr), {"text": prompt}], AXIS_SCHEMA, temperature=0.0, thinking=THINKING)
    centre, end_a, end_b = (_to_pixels(a.get(k), image_bgr) for k in ("centre", "end_a", "end_b"))
    if not centre or list(a.get("centre", [0, 0])) == [0, 0]:
        return None
    long_side = (end_a, end_b) if a.get("elongated") and end_a and end_b and end_a != end_b else None
    return {"x": centre[0], "y": centre[1], "long_side": long_side}


def list_items(image: np.ndarray | str | Path, model: str = config.POINT_MODEL) -> list[dict]:
    """List every loose object a small gripper could pick up.

    Returns `{"label", "x", "y"}` per object (pixels, at its base). Containers, the table,
    the arm and anything fixed are not listed. Used to count what is left to move.
    """
    image_bgr = load(image)
    prompt = (
        "List every loose, graspable object in the image: small things a robot gripper could pick up, "
        "including ones lying on their side, partly hidden, or inside a container. Do not list containers "
        "(bowls, cups, boxes, trays), the table or floor, a robot arm, or anything fixed in place. "
        "Give one entry per object, even when several look the same. `label` is a short visual description "
        "of two or three words that would pick it out in a photo, such as \"black pawn\" or \"red block\". "
        + BASE_POINT
    )
    answer = gemini.generate(model, [_image_part(image_bgr), {"text": prompt}], ITEMS_SCHEMA, temperature=0.0, thinking=THINKING)
    items = []
    for entry in answer:
        pixels = _to_pixels(entry.get("point"), image_bgr)
        if pixels and entry.get("label"):
            items.append({"label": str(entry["label"]), "x": pixels[0], "y": pixels[1]})
    return items


def check_placed(image: np.ndarray | str | Path, item: str, destination: str,
                 model: str = config.POINT_MODEL) -> dict:
    """Judge from a photo whether `item` is now in or at `destination`.

    Returns `{"ok": bool, "reason": str}`; the reason is one short sentence for the screen.
    """
    image_bgr = load(image)
    prompt = (
        f"A robot was asked to put this item: \"{item}\" at this destination: \"{destination}\".\n"
        "Look at the image and decide whether the item is there now. For a container, the item must be inside "
        "it. For a place described relative to other things, it must be roughly there. If you cannot see the "
        "item or the destination, the answer is no. Give `reason` as one short sentence saying what you see, "
        "then `ok`."
    )
    answer = gemini.generate(model, [_image_part(image_bgr), {"text": prompt}], PLACED_SCHEMA, temperature=0.0, thinking=THINKING)
    return {"ok": bool(answer.get("ok")), "reason": str(answer.get("reason", ""))}


def draw(image: np.ndarray | str | Path, points: list[dict]) -> np.ndarray:
    """Return a copy of the image with a numbered circle and the label at each point."""
    canvas = load(image).copy()
    size = max(canvas.shape[:2]) / 1000          # keeps marks readable at any resolution
    radius, thick = round(9 * size), max(1, round(2 * size))
    for number, point in enumerate(points, 1):
        centre = (round(point["x"]), round(point["y"]))
        cv2.circle(canvas, centre, radius, (255, 255, 255), thick + 2)
        cv2.circle(canvas, centre, radius, (0, 0, 255), thick)
        cv2.circle(canvas, centre, max(1, thick), (0, 0, 255), -1)
        text = f"{number} {point.get('label', '')}".strip()
        origin = (centre[0] + radius + 4, centre[1] + radius // 2)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * size, (255, 255, 255), thick + 2, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * size, (0, 0, 255), thick, cv2.LINE_AA)
    return canvas
