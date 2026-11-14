"""Earshot's main loop: teach once, then the arm helps or does the task.

    python -m earshot.app              # camera + screen + arm
    python -m earshot.app --no-arm     # camera + screen only; picks are announced, not made

One thread reads actions (voice, taps, on-screen buttons) and edits the task; a second
thread runs the robot job, checking between moves whether it was paused, changed or stopped.
The arm only looks at the table while it is back at rest, so it never hides what it is finding.
"""
from __future__ import annotations

import argparse
import threading
import time
from typing import Optional

import cv2
import numpy as np

from earshot import config, store
from earshot.camera import Camera
from earshot.schema import Step, Task
from earshot.state import state
from earshot.ui import server
from earshot.vision import gemini, point, watch

TALL = ("cup", "glass", "mug", "bottle")           # containers that need the high, leaning drop
RECORD_FPS = 5


class Earshot:
    def __init__(self, use_arm: bool, speed: float):
        self.cam = Camera().start()
        self.arm = None
        if use_arm:
            from earshot.arm.motion import Arm
            self.arm = Arm(speed=speed)
        self.table = self.arm.table if self.arm else self._table_or_none()
        self.job: Optional[threading.Thread] = None
        self.stop_job = threading.Event()
        self.paused = threading.Event()
        self.recording: Optional[dict] = None       # {"name", "frames", "words", "t0"}
        self.handover_ref: Optional[np.ndarray] = None

    @staticmethod
    def _table_or_none():
        try:
            from earshot.arm.table import Table
            return Table.load()
        except Exception:
            return None

    # ------------------------------------------------------------------ teaching
    def start_teaching(self, name: str = "") -> None:
        if self.recording:
            return
        self._end_job()
        self.recording = {"name": name or time.strftime("task %H:%M"), "frames": [], "words": [], "t0": time.time()}
        state.set_task(None)
        state.set_mode("teaching")
        state.robot("Watching. Do the task once and say what you are doing.")
        threading.Thread(target=self._record, daemon=True).start()

    def _record(self) -> None:
        rec = self.recording
        while self.recording is rec and rec is not None:
            rec["frames"].append(self.cam.jpeg())
            time.sleep(1.0 / RECORD_FPS)

    def stop_teaching(self) -> None:
        rec, self.recording = self.recording, None
        if not rec or len(rec["frames"]) < RECORD_FPS:
            state.set_mode("idle")
            return
        state.set_mode("understanding")
        state.robot("Working out what you did...")
        threading.Thread(target=self._understand, args=(rec,), daemon=True).start()

    def _understand(self, rec: dict) -> None:
        try:
            clip = config.DATA / "clips" / f"{time.strftime('%H%M%S')}.mp4"
            clip.parent.mkdir(parents=True, exist_ok=True)
            first = cv2.imdecode(np.frombuffer(rec["frames"][0], np.uint8), cv2.IMREAD_COLOR)
            out = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), RECORD_FPS, (first.shape[1], first.shape[0]))
            for jpg in rec["frames"]:
                out.write(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR))
            out.release()
            words = " ".join(rec["words"])
            task = watch.watch_clip(clip, words, rec["name"])      # "" = nothing was said
            task.task_id = store.new_task_id(task.name, rec["t0"])
            task.taught_at, task.clip, task.transcript = rec["t0"], str(clip), words
            store.save_task(task)
            state.set_task(task)
            for s in task.steps:
                if not s.agrees:
                    state.check(f"Step {s.n}: picture and words disagree; using the picture.")
            state.robot(f"Learned \"{task.name}\": {len(task.steps)} steps" +
                        (f", {len(task.rules)} rule(s)." if task.rules else "."))
        except Exception as e:                                       # shown on the screen, not swallowed
            state.error(f"Could not understand the demonstration: {e}")
        state.set_mode("idle")

    # ------------------------------------------------------------------ robot jobs
    def _start_job(self, fn, mode: str) -> None:
        if not state.task:
            state.error("Nothing has been taught yet.")
            return
        self._end_job()
        self.stop_job.clear()
        self.paused.clear()
        if self.arm:
            state.check(self.arm.follow_camera(self.cam.fresh_frame()))
            self.table = self.arm.table
        state.reset_steps()
        state.set_mode(mode)
        self.job = threading.Thread(target=self._run_job, args=(fn,), daemon=True)
        self.job.start()

    def _run_job(self, fn) -> None:
        try:
            fn()
        except Exception as e:
            state.error(str(e))
        if not self.stop_job.is_set():
            state.set_mode("idle")

    def _end_job(self) -> None:
        if self.job and self.job.is_alive():
            self.stop_job.set()
            self.paused.clear()
            self.job.join(timeout=60)
        self.job = None

    def _gate(self) -> bool:
        """Between moves: wait while paused; False means the job was stopped."""
        while self.paused.is_set() and not self.stop_job.is_set():
            time.sleep(0.1)
        return not self.stop_job.is_set()

    def _next_todo(self) -> Optional[Step]:
        task = state.task
        return next((s for s in task.steps if s.status == "todo"), None) if task else None

    def _grasp(self, what: str) -> Optional[dict]:
        """Where an item is and which way it lies, for picking it up."""
        return point.grasp_of(self.cam.fresh_frame(), what)

    def _find(self, what: str) -> Optional[tuple[float, float]]:
        hits = point.point_at(self.cam.fresh_frame(), [what])
        return (hits[0]["x"], hits[0]["y"]) if hits else None

    def _move(self, pick_px, place_px, drop: bool, tall: bool = False, long_side=None) -> bool:
        """False when the gripper closed on nothing."""
        if self.arm:
            return self.arm.pick_place(pick_px, place_px, drop=drop, tall=tall, long_side_px=long_side)
        state.robot(f"(no arm) would move {tuple(round(v) for v in pick_px)} to {tuple(round(v) for v in place_px)}")
        time.sleep(1.0)
        return True

    def _do_step(self, step: Step, place_px, drop: bool, verify) -> bool:
        """Pick the step's item and put it at place_px; check with the camera; retry once."""
        for attempt in (1, 2):
            if not self._gate():
                return False
            grasp = self._grasp(step.item)
            if grasp is None:
                state.check(f"Cannot see {step.item} on the table.")
                return False
            pick_px = (grasp["x"], grasp["y"])
            state.pick_attempted()
            state.robot(f"Step {step.n}: {step.item} to {step.destination}" + (" (second try)" if attempt == 2 else ""))
            held = self._move(pick_px, place_px, drop, tall=any(t in step.destination.lower() for t in TALL),
                              long_side=grasp["long_side"])
            ok, why = verify() if held else (False, "the gripper closed on nothing; looking again")
            if ok:
                state.pick_succeeded()
                state.check(f"Step {step.n} checked: {why}")
                return True
            state.check(f"Step {step.n} not right: {why}")
            if attempt == 1:
                state.pick_retried()
        return False

    def _destination_px(self, step: Step, frame) -> Optional[tuple[float, float]]:
        if step.dest_kind == "place" and step.place_xy:
            h, w = frame.shape[:2]
            return step.place_xy[0] * w / 1000.0, step.place_xy[1] * h / 1000.0
        return self._find(f"the middle of the opening of the {step.destination}" if step.dest_kind == "container"
                          else step.destination)

    def do_alone(self) -> None:
        """Run every step, then apply any sorting rule to whatever is still loose."""
        while (step := self._next_todo()) is not None:
            if not self._gate():
                return
            state.set_step_status(step.n, "doing")
            place_px = self._destination_px(step, self.cam.fresh_frame())
            if place_px is None:
                state.check(f"Cannot see {step.destination}.")
                state.set_step_status(step.n, "failed")
                continue
            done = self._do_step(step, place_px, drop=step.dest_kind == "container",
                                 verify=lambda s=step: self._placed(s.item, s.destination))
            state.set_step_status(step.n, "done" if done else "failed")
        self._apply_rules()
        state.robot("Finished.")

    def _placed(self, item: str, destination: str) -> tuple[bool, str]:
        r = point.check_placed(self.cam.fresh_frame(), item, destination)
        return bool(r.get("ok")), str(r.get("reason", ""))

    def _apply_rules(self) -> None:
        task = state.task
        if not task or not task.rules or not self._gate():
            return
        loose = [i for i in point.list_items(self.cam.fresh_frame())]
        if not loose:
            return
        rules = "; ".join(f"if {r.when} -> {r.destination}" for r in task.rules)
        labels = [i["label"] for i in loose]
        plan = gemini.generate(config.WATCH_MODEL, [{"text":
            f"Sorting rules: {rules}.\nLoose items on the table: {labels}.\nFor each item that a rule covers, give its "
            "destination. Leave out items no rule covers, and anything that is itself a container or a destination."}],
            json_schema={"type": "array", "items": {"type": "object", "properties": {
                "item": {"type": "string"}, "destination": {"type": "string"}, "because": {"type": "string"}},
                "required": ["item", "destination", "because"]}})
        n = len(task.steps)
        for p in plan or []:
            n += 1
            step = Step(n=n, item=p["item"], destination=p["destination"], dest_kind="container",
                        seen="Not in the demonstration", heard=f"Rule: {p['because']}")
            state.update_task(lambda t, s=step: t.steps.append(s))
            state.set_step_status(step.n, "doing")
            place_px = self._destination_px(step, self.cam.fresh_frame())
            done = place_px is not None and self._do_step(step, place_px, drop=True,
                                                          verify=lambda s=step: self._placed(s.item, s.destination))
            state.set_step_status(step.n, "done" if done else "failed")

    def help(self) -> None:
        """Hand over the next item each time the person takes the last one."""
        spot = self._handover_px()
        self.handover_ref = self._patch(self.cam.fresh_frame(), spot)
        while (step := self._next_todo()) is not None:
            if not self._gate():
                return
            state.set_step_status(step.n, "doing")
            done = self._do_step(step, spot, drop=False, verify=lambda: self._spot_state(spot, want_item=True))
            if not done:
                state.set_step_status(step.n, "failed")
                continue
            state.robot(f"{step.item} is ready for you. It goes {step.destination}.")
            while self._gate() and self._spot_state(spot, want_item=False)[0] is False:
                time.sleep(0.3)                       # wait until the person takes it
            state.set_step_status(step.n, "done")
        state.robot("That was the last one.")

    def _handover_px(self) -> tuple[float, float]:
        if self.table is None:
            return config.CAMERA_SIZE[0] * 0.5, config.CAMERA_SIZE[1] * 0.75
        return self.table.to_pixel(*config.HANDOVER_XY)

    @staticmethod
    def _patch(frame, px, r: int = 45) -> np.ndarray:
        u, v = int(px[0]), int(px[1])
        return cv2.GaussianBlur(frame[max(0, v - r): v + r, max(0, u - r): u + r], (5, 5), 0).astype(np.int16)

    def _spot_state(self, spot, want_item: bool) -> tuple[bool, str]:
        """Compare the handover spot with how it looked empty."""
        now = self._patch(self.cam.fresh_frame(), spot)
        changed = float(np.mean(np.abs(now - self.handover_ref))) > 12.0
        if want_item:
            return changed, "the item is on the handover spot" if changed else "the handover spot is still empty"
        return (not changed), "taken" if not changed else "still there"

    # ------------------------------------------------------------------ actions
    def handle(self, a: dict) -> None:
        act, args, said = a["action"], a.get("args", {}), a.get("said", "")
        task = state.task
        if act == "transcript":
            state.set_transcript(args.get("text", ""), final=True)
            if self.recording:
                self.recording["words"].append(args.get("text", ""))
        elif act in ("start_teaching",):
            self.start_teaching(args.get("name", ""))
        elif act == "stop_teaching":
            self.stop_teaching()
        elif act in ("tap", "double_tap"):
            self.stop_teaching() if self.recording else self.start_teaching()
        elif act == "help_me":
            self._start_job(self.help, "helping")
        elif act == "your_turn":
            self._start_job(self.do_alone, "doing")
        elif act == "pause":
            self.paused.set()
            state.pause()
        elif act == "resume":
            self.paused.clear()
            state.resume()
        elif act == "load_task":
            found = store.find_task(args.get("query", "the last one"), time.time())
            if found:
                self._end_job()
                state.set_task(found)
                state.robot(f"Loaded \"{found.name}\", taught at {time.strftime('%H:%M', time.localtime(found.taught_at))}.")
            else:
                state.error(f"No saved task matches \"{args.get('query', '')}\".")
        elif act == "swap_steps" and task:
            a_n, b_n = self._step_no(args["a"]), self._step_no(args["b"])
            if a_n and b_n:
                state.swap_steps(a_n, b_n)
                self._corrected(said, f"swapped steps {a_n} and {b_n}")
        elif act == "move_step" and task:
            s_n, before = self._step_no(args["step"]), self._step_no(args["before"])
            if s_n and before:
                state.move_step(s_n, before)
                self._corrected(said, f"moved step {s_n} before step {before}")
        elif act == "change_destination" and task:
            n, dest = self._step_no(args["step_or_item"]), args["destination"]
            hit = next((s for s in task.steps if s.n == n), None)
            if hit:
                state.update_task(lambda t: setattr(hit, "destination", dest))
                self._corrected(said, f"{hit.item} now goes to {dest}")
        elif act == "set_grip":
            self._setting("grip", +0.2 if args.get("level") == "firmer" else -0.2, said)
        elif act == "set_speed":
            self._setting("speed", +0.2 if args.get("level") == "faster" else -0.2, said)
        elif act in ("retry_last", "skip_step") and task:
            last = next((s for s in reversed(task.steps) if s.status in ("failed", "done", "doing")), None)
            if act == "retry_last" and last:
                state.set_step_status(last.n, "todo")
                self._corrected(said, f"retrying step {last.n}")
            elif act == "skip_step" and (nxt := self._next_todo()):
                state.set_step_status(nxt.n, "done")
                self._corrected(said, f"skipped step {nxt.n}")

    def _step_no(self, ref) -> Optional[int]:
        """Resolve "3", "the third one" or "the yellow one" to a step number in the current task."""
        task, text = state.task, str(ref).lower().strip()
        if not task:
            return None
        if text.isdigit():
            return int(text) if 1 <= int(text) <= len(task.steps) else None
        ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
        for i, word in enumerate(ordinals, 1):
            if word in text.split() and i <= len(task.steps):
                return i
        if "last" in text.split():
            return len(task.steps)
        words = {w for w in text.replace(",", " ").split() if w not in ("the", "one", "a", "piece", "block", "item", "step")}
        best = max(task.steps, key=lambda s: len(words & set(s.item.lower().split())), default=None)
        if best and words & set(best.item.lower().split()):
            return best.n
        state.error(f"Not sure which step \"{ref}\" means.")
        return None

    def _corrected(self, said: str, effect: str) -> None:
        state.add_correction(said or "(button)", effect)
        if state.task:
            store.save_task(state.task)

    def _setting(self, name: str, delta: float, said: str) -> None:
        if not self.arm:
            return
        value = float(np.clip(getattr(self.arm, name) + delta, 0.1, 1.0))
        setattr(self.arm, name, value)
        if state.task:
            state.update_task(lambda t: t.settings.__setitem__(name, value))
        self._corrected(said, f"{name} set to {value:.1f}")

    def run(self) -> None:
        server.set_frame_provider(self.cam.jpeg)
        print("Screen:", server.run())
        state.robot("Ready. Say \"watch me\" or tap the band to teach a task.")
        try:
            while True:
                a = server.next_voice_action(timeout=0.5)
                if a:
                    try:
                        self.handle(a)
                    except Exception as e:
                        state.error(f"{a['action']}: {e}")
        except KeyboardInterrupt:
            pass
        finally:
            self._end_job()
            self.cam.stop()
            if self.arm:
                self.arm.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-arm", action="store_true")
    ap.add_argument("--speed", type=float, default=0.4)
    args = ap.parse_args()
    Earshot(use_arm=not args.no_arm, speed=args.speed).run()
