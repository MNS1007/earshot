"""Play a scripted, fake session on the live screen so it can be reviewed with no hardware.

    python scripts/ui_demo.py              # loops until Ctrl-C
    python scripts/ui_demo.py --once       # play once, then keep serving the final screen
    python scripts/ui_demo.py --speed 2    # twice as fast

Nothing here touches a camera, microphone or the arm. The picture is a still from
data/fixtures (or the built-in placeholder if that file is missing). Everything on screen is
driven through earshot.state.state, exactly as the real arm/vision/voice code will do it.
The Pause and Resume buttons work; the other buttons are only logged.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from earshot import config  # noqa: E402
from earshot.schema import Rule, Step, Task  # noqa: E402
from earshot.state import state  # noqa: E402
from earshot.ui import server  # noqa: E402

SPEED = 1.0


def wait(seconds: float) -> None:
    """Sleep, but keep reading the action queue so Pause / Resume on the screen work."""
    end = time.time() + seconds / SPEED
    while True:
        act = server.next_voice_action(timeout=0.05)
        if act and act["action"] == "pause" and state.mode != "paused":
            was = state.snapshot()["robot_status"]
            state.pause()
            state.set_robot_status("Stopped")
            started = time.time()
            while True:
                act = server.next_voice_action(timeout=0.1)
                if act and act["action"] == "resume":
                    break
            state.resume()
            state.set_robot_status(was)
            end += time.time() - started
        if time.time() >= end:
            return


def say(text: str, pace: float = 0.16) -> None:
    """Show a line arriving word by word, as the band transcribes it."""
    words = text.split()
    for i in range(1, len(words) + 1):
        state.set_transcript(" ".join(words[:i]), final=False)
        wait(pace)
    state.set_transcript(text, final=True)


def demo_task() -> Task:
    steps = [
        Step(1, "red block", "left end of the row", "place", [180, 520],
             seen="red block set down at the left edge of the mat", heard="first the red one goes on the left"),
        Step(2, "blue block", "right of the red block", "place", [300, 520],
             seen="blue block placed next to the red block", heard="then blue next to it"),
        Step(3, "yellow block", "right of the blue block", "place", [420, 520],
             seen="yellow block placed third in the row", heard="yellow after that"),
        Step(4, "green block", "right of the yellow block", "place", [540, 520],
             seen="green block placed fourth in the row", heard="then the orange one", agrees=False),
        Step(5, "orange block", "right end of the row", "place", [660, 520],
             seen="orange block placed last in the row", heard="and orange... no, that was green, orange is last"),
        Step(6, "black piece", "cup", "container", None,
             seen="black piece dropped into the cup", heard="anything black just goes in the cup"),
    ]
    return Task(task_id="demo-colour-row", name="Colour row", kind="arrange", taught_at=time.time(),
                steps=steps, rules=[Rule(when="black pieces", destination="cup")],
                settings={"grip": 0.6, "speed": 0.5}, clip="data/fixtures/pattern.mp4")


def pick(n: int, fail_first: bool = False, hand_over: bool = False) -> None:
    """One step: reach, (optionally fail the check and retry), confirm."""
    step = next(s for s in state.task.steps if s.n == n)
    where = "your hand" if hand_over else step.destination
    state.set_step_status(n, "doing")
    state.pick_attempted()
    state.robot(f"Step {n}: picking the {step.item}")
    wait(1.6)
    if fail_first:
        state.check(f"Step {n}: the {step.item} is still on the table. Gripper closed on nothing.")
        state.set_step_status(n, "failed")
        wait(1.6)
        state.pick_retried()
        state.set_step_status(n, "doing")
        state.robot(f"Step {n}: trying the {step.item} again")
        wait(1.8)
    state.robot(f"Step {n}: taking the {step.item} to {where}")
    wait(1.4)
    state.check(f"Step {n}: {step.item} is at {where}. OK.")
    state.pick_succeeded()
    state.set_step_status(n, "done")
    wait(0.7)


def play() -> None:
    state.reset()
    state.set_robot_status("Ready")
    wait(2.5)

    # 1. teaching ---------------------------------------------------------------
    say("watch me")
    state.heard("“watch me” → start_teaching")
    state.set_mode("teaching")
    state.set_robot_status("Arm parked, out of your way")
    wait(1.0)
    for line, saw in [
        ("first the red one goes on the left", "hand places a red block at the left"),
        ("then blue next to it", "blue block placed beside the red block"),
        ("yellow after that", "yellow block placed third"),
        ("then the orange one", "a GREEN block placed fourth"),
        ("and orange... no, that was green, orange is last", "orange block placed last"),
        ("anything black just goes in the cup", "black piece dropped into the cup"),
    ]:
        say(line)
        state.seen(saw)
        wait(0.9)
    say("that's it")
    state.heard("“that's it” → stop_teaching")

    # 2. understanding ----------------------------------------------------------
    state.set_mode("understanding")
    state.set_robot_status("Watching the clip again")
    wait(1.5)
    state.seen("clip: 31 s, 6 moves found")
    wait(1.5)
    task = demo_task()
    state.set_task(task)
    state.check("Step 4: heard “orange”, saw a green block. Picture and words disagree; using the picture.")
    state.set_robot_status("6 steps and 1 rule learned")
    wait(5.0)

    # 3. helping ----------------------------------------------------------------
    say("help me")
    state.heard("“help me” → help_me")
    state.set_mode("helping")
    wait(0.8)
    pick(1, fail_first=True, hand_over=True)

    state.set_step_status(2, "doing")
    state.pick_attempted()
    state.robot("Step 2: reaching for the blue block")
    wait(1.2)
    say("no, yellow before blue")
    state.heard("“no, yellow before blue” → reorder")
    state.set_robot_status("Stopped. Changing the order")
    state.set_step_status(2, "todo")
    wait(0.8)

    def reorder(t: Task) -> None:
        by_item = {s.item: s for s in t.steps}
        by_item["yellow block"].destination = "right of the red block"
        by_item["blue block"].destination = "right of the yellow block"
        by_item["green block"].destination = "right of the blue block"
    state.move_step(3, before=2)
    state.update_task(reorder)
    state.add_correction("yellow before blue", "moved step 3 before step 2")
    wait(3.0)

    state.set_step_status(2, "doing")          # same attempt, now aimed at the yellow block
    state.robot("Step 2: picking the yellow block")
    wait(1.6)
    state.robot("Step 2: taking the yellow block to your hand")
    wait(1.4)
    state.check("Step 2: yellow block is at your hand. OK.")
    state.pick_succeeded()
    state.set_step_status(2, "done")
    wait(0.7)
    pick(3, hand_over=True)

    # 4. doing ------------------------------------------------------------------
    say("okay, your turn, do it yourself")
    state.heard("“your turn” → your_turn")
    state.set_mode("doing")
    state.reset_steps()
    state.set_robot_status("Starting again from step 1, on my own")
    wait(2.0)
    for n in range(1, 7):
        pick(n, fail_first=(n == 5))
    state.set_transcript("", final=True)
    state.robot("Finished: 6 of 6 in place")
    wait(8.0)


def main() -> None:
    global SPEED
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--once", action="store_true", help="play once, then keep serving the final screen")
    ap.add_argument("--speed", type=float, default=1.0, help="2 = twice as fast")
    ap.add_argument("--port", type=int, default=config.UI_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    SPEED = max(0.1, args.speed)

    still = config.DATA / "fixtures" / "pile.jpg"
    if still.exists():
        jpeg = still.read_bytes()
        server.set_frame_provider(lambda: jpeg)

    url = server.run(host=args.host, port=args.port)
    print(f"Earshot screen: {url}   (Ctrl-C to stop)", flush=True)
    try:
        while True:
            play()
            if args.once:
                state.set_mode("idle")
                print("Demo finished; still serving the final screen. Ctrl-C to stop.", flush=True)
                while True:
                    wait(3600)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
