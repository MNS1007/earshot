"""Saved tasks: one JSON file per Task under config.TASKS_DIR, recalled by name or by time.

    save_task(task)              write/overwrite data/tasks/<task_id>.json
    load_task(task_id)           read one back
    list_tasks()                 every saved task, newest first
    find_task(query, now)        resolve a spoken reference: "the sorting one", "what I taught
                                 at 10", "the one from this morning", "the last one"

Pure Python, no I/O other than the JSON files. `find_task` never reads the clock; the caller
passes `now` (seconds since the epoch) so it can be tested with fixed times.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from . import config
from .schema import Task

__all__ = ["save_task", "load_task", "list_tasks", "find_task", "new_task_id"]


# ----------------------------------------------------------------------------- files

def _dir() -> Path:
    # Read at call time so tests (and the venue) can point config.TASKS_DIR elsewhere.
    return Path(config.TASKS_DIR)


def _safe_id(task_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id)).strip("._")
    if not safe:
        raise ValueError(f"unusable task_id: {task_id!r}")
    return safe


def _path(task_id: str) -> Path:
    return _dir() / f"{_safe_id(task_id)}.json"


def new_task_id(name: str, now: float) -> str:
    """A readable, sortable id: '20261114-1042-colour-row'."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "task"
    return f"{datetime.fromtimestamp(now):%Y%m%d-%H%M}-{slug[:40]}"


def save_task(task: Task) -> Path:
    """Write the task as JSON (atomically) and return the file path."""
    path = _path(task.task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(task.to_json(), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_task(task_id: str) -> Task:
    """Read one task. Raises FileNotFoundError if it was never saved."""
    return Task.from_json(_path(task_id).read_text(encoding="utf-8"))


def list_tasks() -> list[Task]:
    """All saved tasks, newest first. Unreadable files are skipped, not fatal."""
    d = _dir()
    if not d.is_dir():
        return []
    tasks: list[Task] = []
    for f in d.glob("*.json"):
        try:
            tasks.append(Task.from_json(f.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    tasks.sort(key=lambda t: t.taught_at, reverse=True)
    return tasks


# ----------------------------------------------------------------------------- spoken references

_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_MIN = {"thirty": 30, "fifteen": 15, "forty five": 45, "forty-five": 45, "twenty": 20, "ten": 10,
        "oh five": 5, "forty": 40, "fifty": 50}
_NUM_RE = "|".join(_NUM)
_MIN_RE = "|".join(sorted(_MIN, key=len, reverse=True))
_AMPM = r"(am|pm)"

# Tried in order; the first that matches wins. Groups: hour, minute, am/pm (each optional by name).
_CLOCK_PATTERNS = [
    re.compile(rf"\bhalf past (?P<h>\d{{1,2}}|{_NUM_RE})\b(?P<half>)\s*(?P<ap>am|pm)?"),
    re.compile(rf"\b(?P<h>\d{{1,2}})[:.](?P<m>\d{{2}})\s*(?P<ap>am|pm)?\b"),
    re.compile(rf"\b(?P<h>\d{{1,2}})\s*(?P<ap>am|pm)\b"),
    re.compile(rf"\b(?P<h>\d{{1,2}}|{_NUM_RE})\s+o'?\s?clock\b"),
    re.compile(rf"\b(?:at|around|about|near)\s+(?P<h>\d{{1,2}})\b(?!\s*(?:steps?|blocks?|items?|pieces?))"),
    re.compile(rf"\b(?P<h>{_NUM_RE})(?:\s+(?P<mw>{_MIN_RE}))?\s*(?P<ap>am|pm)\b"),
    re.compile(rf"\b(?:at|around|about|near)\s+(?P<h>{_NUM_RE})(?:\s+(?P<mw>{_MIN_RE}))?\b"),
    re.compile(r"\b(?P<noon>noon|midday)\b"),
    re.compile(r"\b(?P<midnight>midnight)\b"),
]

_PARTS = [  # (pattern, start hour, end hour)
    (re.compile(r"\bmorning\b"), 0, 12),
    (re.compile(r"\bafternoon\b"), 12, 18),
    (re.compile(r"\b(evening|tonight|night)\b"), 17, 24),
]

_SECOND_LAST = re.compile(r"\b(second (to )?last|(one )?before (the )?last|before that( one)?|last but one)\b")
_LAST = re.compile(r"\b(last|latest|newest|most recent|recent|previous|just now|just taught|just showed)\b")
_FIRST = re.compile(r"\b(first|earliest|oldest)\b")

_STOP = set("""
the a an one ones task tasks thing that this these those what which who i you we me my our your it its
taught teach teaching showed show shown did do does done made make from at of on in to into by with for
was is are were be been load open run rerun redo repeat again please can could would will lets let us s
earlier before ago time about around near o clock oclock am pm today yesterday remember when where had
have has just and or robot back go bring up get use using start same as like them job demo then now
called named say said
""".split())

_CLOCK_WINDOW = 60 * 60       # "at 10" accepts a task taught within an hour of 10:00
_FUTURE_SLACK = 30 * 60       # at 9:50, "at 10" may still mean today's 10:00


def _normalise(query: str) -> str:
    q = query.lower().replace("’", "'")
    q = re.sub(r"\ba\.?\s?m\b\.?", "am", q)
    q = re.sub(r"\bp\.?\s?m\b\.?", "pm", q)
    q = re.sub(r"[^a-z0-9:.' -]+", " ", q)
    q = re.sub(r"\.(?!\d)", " ", q)          # keep 10.30, drop sentence full stops
    return re.sub(r"\s+", " ", q).strip()


def _parse_clock(q: str) -> tuple[Optional[tuple[int, int, Optional[str]]], str]:
    """Find a clock time. Returns ((hour, minute, 'am'|'pm'|None) or None, query with it removed)."""
    for pat in _CLOCK_PATTERNS:
        m = pat.search(q)
        if not m:
            continue
        g = m.groupdict()
        if g.get("noon"):
            h, mi, ap = 12, 0, "pm"
        elif g.get("midnight"):
            h, mi, ap = 0, 0, "am"
        else:
            raw = g["h"]
            h = int(raw) if raw.isdigit() else _NUM[raw]
            mi = int(g["m"]) if g.get("m") else _MIN.get(g.get("mw") or "", 0)
            if "half" in g and g["half"] is not None:
                mi = 30
            ap = g.get("ap")
        if h > 23 or mi > 59:
            continue
        return (h, mi, ap), (q[:m.start()] + " " + q[m.end():]).strip()
    return None, q


def _hours(h: int, ap: Optional[str], part: Optional[tuple[int, int]]) -> list[int]:
    """The 24-hour readings a spoken hour could mean."""
    if ap == "am":
        return [h % 12]
    if ap == "pm":
        return [h % 12 + 12]
    if h == 0 or h > 12:
        return [h]
    both = [h % 12, h % 12 + 12]
    if part:
        inside = [x for x in both if part[0] <= x < part[1] or (x == 12 and part[1] == 12)]
        return inside or both
    return both


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word[:-1] if word.endswith("e") and len(word) > 3 else word


def _word_match(a: str, b: str) -> bool:
    a, b = _stem(a), _stem(b)
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 3 and long_.startswith(short):
        return True
    return len(short) >= 5 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]


def _name_score(tokens: list[str], task: Task) -> int:
    strong = _words(task.name) + _words(task.kind) + [w for w in _words(task.task_id) if not w.isdigit()]
    weak: list[str] = []
    for s in task.steps:
        weak += _words(s.item) + _words(s.destination)
    for r in task.rules:
        weak += _words(r.when) + _words(r.destination)
    weak = [w for w in weak if w not in _STOP]
    score = 0
    for tok in tokens:
        if any(_word_match(tok, w) for w in strong):
            score += 3
        elif any(_word_match(tok, w) for w in weak):
            score += 1
    return score


def _best_by_name(tokens: list[str], pool: list[Task]) -> Optional[Task]:
    """Highest name score; ties go to the newer task (pool is newest first)."""
    best, best_score = None, 0
    for t in pool:
        s = _name_score(tokens, t)
        if s > best_score:
            best, best_score = t, s
    return best


def find_task(query: str, now: float, tasks: Optional[Iterable[Task]] = None) -> Optional[Task]:
    """Resolve a spoken reference to a saved task, or None when nothing fits.

    By time:  "the last one", "the one before that", "the first one", "the one from this morning",
              "yesterday afternoon", "what I taught at 10", "at 10:30", "at ten", "2 pm".
              A bare hour means the most recent 10 o'clock that is not in the future; the task
              taught nearest to it (within an hour) is returned.
    By name:  "the sorting one", "blocks", "colour row" -- matched on the task's name and kind
              first, then on its items and destinations. Small speech-to-text slips are tolerated.
    Both:     "the sorting one from this morning" narrows by time, then by name.

    `now` is seconds since the epoch (local time is used for clock phrases). `tasks` defaults to
    list_tasks(); pass a list to test without touching disk.
    """
    pool = sorted(list_tasks() if tasks is None else tasks, key=lambda t: t.taught_at, reverse=True)
    q = _normalise(query or "")
    if not pool or not q:
        return None
    now_dt = datetime.fromtimestamp(now)
    midnight = now_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    # --- pull the time phrases out of the sentence; what is left may be a name
    day: Optional[int] = None
    if re.search(r"\byesterday\b", q):
        day = -1
    elif re.search(r"\b(today|this (morning|afternoon|evening)|tonight)\b", q):
        day = 0
    part: Optional[tuple[int, int]] = None
    for pat, start, end in _PARTS:
        if pat.search(q):
            part = (start, end)
            q = pat.sub(" ", q)
            break
    clock, q = _parse_clock(q)
    which = None
    for name, pat in (("second", _SECOND_LAST), ("last", _LAST), ("first", _FIRST)):
        if pat.search(q):
            which = name
            q = pat.sub(" ", q)
            break
    tokens = [w for w in _words(q) if w not in _STOP and not w.isdigit()]

    def pick(candidates: list[Task], strict_name: bool) -> Optional[Task]:
        """candidates is newest first and already narrowed by time."""
        if not candidates:
            return None
        if tokens:
            named = _best_by_name(tokens, candidates)
            if named or strict_name:
                return named
        if which == "first":
            return candidates[-1]
        if which == "second":
            return candidates[1] if len(candidates) > 1 else None
        return candidates[0]

    # --- "at 10": nearest task to the most recent plausible 10 o'clock
    if clock:
        h, mi, ap = clock
        targets = []
        for d in ([day] if day is not None else [0, -1]):
            for hour in _hours(h, ap, part):
                targets.append(midnight + timedelta(days=d, hours=hour, minutes=mi))
        targets.sort(reverse=True)
        for target in targets:
            ts = target.timestamp()
            if ts > now + _FUTURE_SLACK:
                continue
            near = [t for t in pool if abs(t.taught_at - ts) <= _CLOCK_WINDOW]
            if not near:
                continue
            if tokens:
                named = _best_by_name(tokens, near)
                if named:
                    return named
            return min(near, key=lambda t: abs(t.taught_at - ts))
        return None

    # --- "this morning", "yesterday", "yesterday afternoon"
    if day is not None or part:
        base = midnight + timedelta(days=day or 0)
        start, end = part or (0, 24)
        lo, hi = (base + timedelta(hours=start)).timestamp(), (base + timedelta(hours=end)).timestamp()
        return pick([t for t in pool if lo <= t.taught_at < hi], strict_name=False)

    # --- "the last one", "the last sorting one", "the first one"
    if which:
        past = [t for t in pool if t.taught_at <= now] or pool
        return pick(past, strict_name=False)

    # --- name only: must actually match something
    if tokens:
        return _best_by_name(tokens, pool)
    return None
