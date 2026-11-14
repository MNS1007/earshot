# Watch one demonstration clip (plus what the wristband heard) and turn it into a Task.
"""Watching: a video model reads the demonstration and lists the moves in order.

In the real system the clip has no usable sound and the words arrive separately as plain
text from the wristband, without timestamps. A clip with its own sound track also works
when no transcript is given.
"""
from __future__ import annotations

import mimetypes
import re
import time
from pathlib import Path

import cv2
import numpy as np

from earshot import config
from earshot.schema import Rule, Step, Task
from earshot.vision import gemini, point

WATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["arrange", "sort", "tidy"]},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "seen": {"type": "string", "description": "what the video shows for this move"},
                    "heard": {"type": "string", "description": "the spoken words for this move, or empty"},
                    "item": {"type": "string"},
                    "destination": {"type": "string"},
                    "dest_kind": {"type": "string", "enum": ["container", "place"]},
                    "place_point": {
                        "type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
                        "description": "only for dest_kind place: [y, x] 0-1000 in the last frame",
                    },
                    "agrees": {"type": "boolean"},
                },
                "required": ["seen", "heard", "item", "destination", "dest_kind", "agrees"],
            },
        },
        "rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"when": {"type": "string"}, "destination": {"type": "string"}},
                "required": ["when", "destination"],
            },
        },
    },
    "required": ["kind", "steps", "rules"],
}

INSTRUCTIONS = """A person is teaching a robot arm a table-top task by doing it once. Watch the whole video and write down what they did so the robot can repeat it.

{words}

Report:
- kind: "arrange" if items are set down in a particular layout, "sort" if items go into different containers or groups according to what they are, "tidy" if everything is simply cleared away into one place.
- steps: one per item the person moves, in the order the moves happen in the video. Count every move, also when two items look the same.
  - seen: one short sentence on what the video shows for this move.
  - heard: the spoken words that go with this move, quoted as given; empty if nothing was said about it.
  - item: a short visual description of two to four words that would pick the object out in a photo of the table, such as "black queen" or "red block". Go by what the video shows. Small objects can be hard to tell apart in the video (a dark pawn and a dark bishop look alike), so when the words give a more exact name that fits what you see (same colour, similar size and shape), use that name and count it as agreeing.
  - destination: where the video shows it ending up. For a container, name the container shortly ("cup", "clear box"). For a spot on the table, describe it relative to the other items or the layout ("far left of the back row", "in front of the black queen").
  - dest_kind: "container" if it is put into or onto something that holds it, otherwise "place".
  - place_point: only for dest_kind "place": the point where this item rests on the table in the LAST frame of the video, as [y, x], each normalised to 0-1000 over the frame (y down from the top, x from the left). Use the middle of the item's base, not its top.
  - agrees: true when the words fit what the video shows for this move, or nothing was said. false when the video clearly shows a different item (another colour or a plainly different object) or a different destination than the words say.
- rules: if the person is sorting by a rule, the rule for each destination, as "when" (a condition on the item, such as "the item is black") and "destination". Include a rule that is never said aloud but is clear from what they did. Empty when the task follows no rule.

The video is the truth. When the words and the video disagree, item and destination must say what the VIDEO shows, heard keeps the words as spoken, and agrees is false. Never copy an item or a destination from the words if the video shows something else."""

WITH_SOUND = "The video has sound. Listen to what the person says while they work and use it as their words."
WITH_TRANSCRIPT = (
    "The video has no usable sound; ignore any audio. The person's words were recorded separately by a "
    "wristband microphone and are given below as plain text without timestamps. They were spoken in order "
    "while working, so match them to the moves by order and by content.\n\nWords from the wristband:\n\"\"\"\n{transcript}\n\"\"\""
)

NO_WORDS = "The video has no usable sound and the person said nothing; ignore any audio and leave every heard empty."


def last_frame(clip_path: str | Path) -> np.ndarray:
    """Return the final readable frame of a clip as a BGR image (the frame place_xy refers to)."""
    capture = cv2.VideoCapture(str(clip_path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for back in range(1, 31):
            capture.set(cv2.CAP_PROP_POS_FRAMES, max(count - back, 0))
            ok, frame = capture.read()
            if ok:
                return frame
    finally:
        capture.release()
    raise ValueError(f"cannot read a frame from {clip_path}")


def _place_xy(point) -> list[float] | None:
    """Convert the model's native [y, x] point to the stored [x, y], both 0-1000."""
    if not isinstance(point, list) or len(point) != 2:
        return None
    y, x = (min(max(float(v), 0.0), 1000.0) for v in point)
    return [x, y]


def _refine_places(steps: list[Step], frame: np.ndarray) -> None:
    """Replace each place step's rough point from the video with one pointed at in the last frame.

    Pointing in a still image is far more exact than a point read off a video. A step the
    pointing model cannot find keeps the video's estimate.
    """
    placed = [s for s in steps if s.dest_kind == "place"]
    if not placed:
        return
    height, width = frame.shape[:2]
    found = point.point_at(frame, [f"{s.item}, {s.destination}" for s in placed])
    for step in placed:
        if found and found[0]["label"] == f"{step.item}, {step.destination}":
            hit = found.pop(0)
            step.place_xy = [round(hit["x"] / width * 1000, 1), round(hit["y"] / height * 1000, 1)]


def watch_clip(clip_path: str | Path, transcript: str | None, name: str) -> Task:
    """Turn one demonstration into a Task.

    `clip_path` is the video of the table. `transcript` is what the wristband heard, as
    plain text; pass None to have the model listen to the clip's own sound track instead
    (an empty string means nothing was said, and the clip's sound is ignored).
    `name` is what the person calls the task. Where words and video disagree, the steps
    follow the video and are marked `agrees=False`. Steps that set an item down on the
    table get `place_xy`, its resting point in the clip's last frame as [x, y] 0-1000.
    """
    clip_path = Path(clip_path)
    spoken = (transcript or "").strip()
    if transcript is None:
        words = WITH_SOUND
    elif spoken:
        words = WITH_TRANSCRIPT.format(transcript=spoken)
    else:
        words = NO_WORDS
    mime = mimetypes.guess_type(clip_path.name)[0] or "video/mp4"
    uploaded = gemini.upload_file(clip_path, mime)
    answer = gemini.generate(
        config.WATCH_MODEL,
        [{"file_data": {"mime_type": uploaded.get("mimeType", mime), "file_uri": uploaded["uri"]}},
         {"text": INSTRUCTIONS.format(words=words)}],
        json_schema=WATCH_SCHEMA,
    )

    steps = []
    for n, raw in enumerate(answer.get("steps", []), 1):
        is_place = raw.get("dest_kind") == "place"
        steps.append(Step(
            n=n,
            item=str(raw.get("item", "")).strip(),
            destination=str(raw.get("destination", "")).strip(),
            dest_kind="place" if is_place else "container",
            place_xy=_place_xy(raw.get("place_point")) if is_place else None,
            seen=str(raw.get("seen", "")).strip(),
            heard=str(raw.get("heard", "")).strip(),
            agrees=bool(raw.get("agrees", True)),
        ))
    _refine_places(steps, last_frame(clip_path))
    rules = [Rule(when=str(r["when"]).strip(), destination=str(r["destination"]).strip())
             for r in answer.get("rules", []) if r.get("when") and r.get("destination")]
    kind = answer.get("kind") if answer.get("kind") in ("arrange", "sort", "tidy") else "arrange"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "task"
    return Task(
        task_id=f"{slug}-{int(time.time())}",
        name=name,
        kind=kind,
        steps=steps,
        rules=rules,
        clip=str(clip_path),
        transcript=spoken,
    )
