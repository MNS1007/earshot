"""The one live state object every part of Earshot writes to and the screen reads from.

    from earshot.state import state

    state.set_mode("teaching")
    state.set_transcript("first the red one goes on the left")
    state.set_task(task)
    state.set_step_status(2, "doing")
    state.robot("Picking the blue block")           # logs an event and sets the robot status line
    state.pick_attempted(); state.pick_succeeded()

Every method takes the lock, changes the state, then tells the subscribers (the UI server pushes
the new state to the browser). Methods are safe to call from any thread.

If you change the Task or its Steps directly (e.g. `task.steps[0].status = "done"`), call
`state.notify()` afterwards, or better, do it inside `state.update_task(fn)`.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict
from typing import Any, Callable, Optional

from .schema import Correction, Rule, Step, Task

__all__ = ["AppState", "state", "MODES", "EVENT_KINDS", "STEP_STATUSES"]

MODES = ("idle", "teaching", "understanding", "helping", "doing", "paused")
EVENT_KINDS = ("heard", "seen", "robot", "check", "correction", "error")
STEP_STATUSES = ("todo", "doing", "done", "failed")

Subscriber = Callable[[dict], None]


class AppState:
    def __init__(self, max_events: int = 200):
        self._lock = threading.RLock()
        self._subs: list[Subscriber] = []
        self._mode = "idle"
        self._paused_from: Optional[str] = None
        self._task: Optional[Task] = None
        self._events: deque[dict] = deque(maxlen=max_events)
        self._transcript = ""
        self._transcript_final = True
        self._transcript_at = 0.0
        self._robot_status = ""
        self._counters = {"attempted": 0, "succeeded": 0, "retried": 0}
        self._version = 0

    # ------------------------------------------------------------------ reading

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def task(self) -> Optional[Task]:
        """The live Task object (not a copy). Call notify() if you change it in place."""
        return self._task

    def snapshot(self) -> dict:
        """A JSON-ready deep copy of everything the screen shows."""
        with self._lock:
            task = asdict(self._task) if self._task else None
            current = None
            if self._task:
                current = next((s.n for s in self._task.steps if s.status == "doing"), None)
            return {
                "version": self._version,
                "now": time.time(),
                "mode": self._mode,
                "paused_from": self._paused_from,
                "task": task,
                "current_step": current,
                "events": list(self._events),            # oldest first
                "transcript": {"text": self._transcript, "final": self._transcript_final,
                               "t": self._transcript_at},
                "robot_status": self._robot_status,
                "counters": dict(self._counters),
            }

    # ------------------------------------------------------------------ subscribers

    def subscribe(self, fn: Subscriber) -> Callable[[], None]:
        """fn(snapshot) is called after every change, on the thread that made the change.
        Returns a function that unsubscribes."""
        with self._lock:
            self._subs.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._subs:
                    self._subs.remove(fn)
        return unsubscribe

    def notify(self) -> None:
        """Tell subscribers the state changed. Called by every setter; call it yourself after
        editing the Task in place."""
        with self._lock:
            self._version += 1
            subs = list(self._subs)
        if not subs:
            return
        snap = self.snapshot()
        for fn in subs:                      # outside the lock: subscribers may call back in
            try:
                fn(snap)
            except Exception:                # a broken screen must never stop the arm
                pass

    # ------------------------------------------------------------------ mode

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        with self._lock:
            if mode == "paused":
                if self._mode != "paused":
                    self._paused_from = self._mode
            else:
                self._paused_from = None
            self._mode = mode
        self.notify()

    def pause(self) -> None:
        self.set_mode("paused")

    def resume(self) -> str:
        """Leave "paused" for whatever mode was running before. Returns the mode now active."""
        with self._lock:
            back = self._paused_from if self._mode == "paused" else self._mode
        self.set_mode(back or "idle")
        return self._mode

    # ------------------------------------------------------------------ task

    def set_task(self, task: Optional[Task]) -> None:
        with self._lock:
            self._task = task
        self.notify()

    def update_task(self, fn: Callable[[Task], Any]) -> Any:
        """Run fn(task) under the lock, then notify. Use for any edit without its own method."""
        with self._lock:
            if self._task is None:
                raise RuntimeError("no current task")
            result = fn(self._task)
        self.notify()
        return result

    def _step(self, n: int) -> Step:
        if self._task is None:
            raise RuntimeError("no current task")
        for s in self._task.steps:
            if s.n == n:
                return s
        raise KeyError(f"no step {n}")

    def set_step_status(self, n: int, status: str) -> None:
        """Mark step n (its current 1-based number) todo / doing / done / failed."""
        if status not in STEP_STATUSES:
            raise ValueError(f"status must be one of {STEP_STATUSES}, got {status!r}")
        with self._lock:
            self._step(n).status = status  # type: ignore[assignment]
        self.notify()

    def reset_steps(self) -> None:
        """Every step back to todo (before running the task again)."""
        with self._lock:
            if self._task:
                for s in self._task.steps:
                    s.status = "todo"
        self.notify()

    def _renumber(self) -> None:
        for i, s in enumerate(self._task.steps, start=1):  # type: ignore[union-attr]
            s.n = i

    def move_step(self, n: int, before: int) -> None:
        """Move step n so it comes just before step `before`; steps are renumbered 1..N."""
        with self._lock:
            step, target = self._step(n), self._step(before)
            if step is not target:
                steps = self._task.steps  # type: ignore[union-attr]
                steps.remove(step)
                steps.insert(steps.index(target), step)
                self._renumber()
        self.notify()

    def swap_steps(self, a: int, b: int) -> None:
        """Exchange the positions of steps a and b; steps are renumbered 1..N."""
        with self._lock:
            steps = self._task.steps if self._task else []
            i, j = steps.index(self._step(a)), steps.index(self._step(b))
            steps[i], steps[j] = steps[j], steps[i]
            self._renumber()
        self.notify()

    def add_rule(self, when: str, destination: str) -> None:
        self.update_task(lambda t: t.rules.append(Rule(when=when, destination=destination)))

    def add_correction(self, said: str, effect: str, at: Optional[float] = None) -> None:
        """Record a spoken correction on the task (shown under the step list) and log it."""
        at = time.time() if at is None else at
        with self._lock:
            if self._task is not None:
                self._task.corrections.append(Correction(at=at, said=said, effect=effect))
            self._events.append({"t": at, "kind": "correction", "text": f"“{said}” → {effect}"})
        self.notify()

    # ------------------------------------------------------------------ events

    def log(self, kind: str, text: str, t: Optional[float] = None) -> None:
        """Append {t, kind, text} to the rolling log. kind: heard/seen/robot/check/correction/error."""
        if kind not in EVENT_KINDS:
            raise ValueError(f"kind must be one of {EVENT_KINDS}, got {kind!r}")
        with self._lock:
            self._events.append({"t": time.time() if t is None else t, "kind": kind, "text": str(text)})
        self.notify()

    def heard(self, text: str) -> None:
        self.log("heard", text)

    def seen(self, text: str) -> None:
        self.log("seen", text)

    def check(self, text: str) -> None:
        self.log("check", text)

    def error(self, text: str) -> None:
        self.log("error", text)

    def robot(self, text: str) -> None:
        """Log a robot event and show the same text as the robot status line."""
        with self._lock:
            self._robot_status = str(text)
        self.log("robot", text)

    def clear_events(self) -> None:
        with self._lock:
            self._events.clear()
        self.notify()

    # ------------------------------------------------------------------ transcript, robot status

    def set_transcript(self, text: str, final: bool = False) -> None:
        """The line the microphone is hearing right now. final=False while the words are still arriving."""
        with self._lock:
            self._transcript = str(text)
            self._transcript_final = bool(final)
            self._transcript_at = time.time()
        self.notify()

    def set_robot_status(self, text: str) -> None:
        """Status line only, without a log entry (use robot() for both)."""
        with self._lock:
            self._robot_status = str(text)
        self.notify()

    # ------------------------------------------------------------------ counters

    def _count(self, name: str, by: int) -> None:
        with self._lock:
            self._counters[name] += by
        self.notify()

    def pick_attempted(self, by: int = 1) -> None:
        """A pick of a new item was started (retries of the same item do not count again)."""
        self._count("attempted", by)

    def pick_succeeded(self, by: int = 1) -> None:
        """The camera check confirmed the item arrived."""
        self._count("succeeded", by)

    def pick_retried(self, by: int = 1) -> None:
        """A pick failed its check and was tried again."""
        self._count("retried", by)

    def reset_counters(self) -> None:
        with self._lock:
            for k in self._counters:
                self._counters[k] = 0
        self.notify()

    def reset(self) -> None:
        """Back to an empty idle screen."""
        with self._lock:
            self._mode, self._paused_from, self._task = "idle", None, None
            self._events.clear()
            self._transcript, self._transcript_final, self._robot_status = "", True, ""
            for k in self._counters:
                self._counters[k] = 0
        self.notify()


state = AppState()
