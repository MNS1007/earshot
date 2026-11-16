"""The task format every part of Earshot agrees on.

A Task is what the robot learned from one demonstration: an ordered list of Steps, optional
sorting Rules, and the spoken Corrections given since. The video model fills it, the screen
shows it, the voice loop edits it, the arm executes it, and the store saves it as JSON.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

Kind = Literal["arrange", "sort", "tidy"]


@dataclass
class Step:
    """One move: pick `item` and put it at `destination`."""
    n: int                                 # 1-based order
    item: str                              # how to name it to the pointing model: "red block"
    destination: str                       # a container ("cup", "box") or a place ("left of the red block")
    dest_kind: Literal["container", "place"] = "container"
    place_xy: Optional[list[float]] = None  # for "place": where it ended up in the demo image, [x, y] 0-1000
    seen: str = ""                         # what the video showed
    heard: str = ""                        # the spoken words that go with it
    agrees: bool = True                    # False when picture and words disagree
    status: Literal["todo", "doing", "done", "failed"] = "todo"


@dataclass
class Rule:
    """A sorting rule that also covers items the person never touched."""
    when: str                              # "the item is red"
    destination: str                       # "bowl"


@dataclass
class Correction:
    """Something the person said afterwards that changes how the task is done."""
    at: float
    said: str
    effect: str                            # plain words: "moved step 3 before step 2"


@dataclass
class Task:
    task_id: str
    name: str
    kind: Kind
    taught_at: float = field(default_factory=time.time)
    steps: list[Step] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    corrections: list[Correction] = field(default_factory=list)
    settings: dict = field(default_factory=dict)   # e.g. {"grip": 0.6, "speed": 0.5}
    clip: str = ""                                 # path of the demonstration clip
    transcript: str = ""                           # what the microphone heard during the demonstration

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(d: dict) -> "Task":
        d = dict(d)
        d["steps"] = [Step(**s) for s in d.get("steps", [])]
        d["rules"] = [Rule(**r) for r in d.get("rules", [])]
        d["corrections"] = [Correction(**c) for c in d.get("corrections", [])]
        return Task(**d)

    @staticmethod
    def from_json(text: str) -> "Task":
        return Task.from_dict(json.loads(text))


# The one arm call the rest of the system uses (implemented in earshot/arm/motion.py):
#
#     pick_place(pick_px: tuple[float, float], place_px: tuple[float, float],
#                drop: bool, grip: float = 0.6) -> bool
#
# pick_px / place_px are pixel coordinates in the table camera's image. `drop=True` releases
# from DROP_Z over a container; `drop=False` sets the item down at GRASP_Z. Returns True when
# the motion finished; whether the item actually arrived is checked afterwards by the camera.
