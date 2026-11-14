"""The action set of the voice loop.

Each function below is one thing a person can ask of the arm. They are Pipecat tool
functions: the LLM picks exactly one per spoken turn, and the function POSTs

    {"action": <name>, "args": {...}, "said": <the words of that turn>}

to the main app (earshot.config.VOICE_POST_URL, or EARSHOT_VOICE_POST_URL when set, which
the tests use). The voice loop never moves the arm and never resolves "the yellow one" to
a step number: it passes the person's own words on, and the main app, which holds the
task, decides what they refer to.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import httpx
from pipecat.frames.frames import FunctionCallResultProperties, TTSSpeakFrame
from pipecat.services.llm_service import FunctionCallParams

from earshot import config

POST_TIMEOUT_S = 2.0


def post_url() -> str:
    return os.environ.get("EARSHOT_VOICE_POST_URL") or config.VOICE_POST_URL


@dataclass
class Sent:
    """One action as it left the voice loop."""
    action: str
    args: dict
    said: str
    delivered: bool            # the main app answered 2xx
    confirm: str               # the short sentence for the speaker ("" = say nothing)
    llm_ms: Optional[float]    # from the start of the LLM request to the tool call
    post_ms: float


@dataclass
class Dispatcher:
    """Shared state of one voice process: the HTTP client, the turn guard, the listeners."""
    speak: bool = False                               # a TTS stage exists in the pipeline
    listeners: list[Callable[[Sent], None]] = field(default_factory=list)
    turn_started: Optional[float] = None              # time.monotonic() when the LLM was last asked
    acted_on: object = None                           # the user message the last action answered
    app_up: Optional[bool] = None                     # None until the first POST
    _client: Optional[httpx.AsyncClient] = None

    def begin_turn(self) -> None:
        self.turn_started = time.monotonic()

    async def post(self, action: str, args: dict, said: str) -> tuple[bool, dict]:
        """POST one action. Never raises; returns (delivered, the app's JSON answer or {})."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=POST_TIMEOUT_S)
        try:
            r = await self._client.post(post_url(), json={"action": action, "args": args, "said": said})
            ok = 200 <= r.status_code < 300
            why = "it answered status %d" % r.status_code
            try:
                answer = r.json() if ok else {}
            except ValueError:
                answer = {}
        except httpx.HTTPError as e:
            ok, why, answer = False, type(e).__name__, {}
        if not ok and self.app_up is not False:       # one line per outage, not one per POST
            print("[voice] The main app is not reachable at %s (%s); actions are dropped until it is back."
                  % (post_url(), why), flush=True)
        elif ok and self.app_up is False:
            print("[voice] The main app is back.", flush=True)
        self.app_up = ok
        return ok, answer if isinstance(answer, dict) else {}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


DISPATCH = Dispatcher()


def last_user_turn(context) -> tuple[str, object]:
    """The words of the newest user message in an LLM context, and that message itself."""
    for m in reversed(list(context.get_messages())):
        if isinstance(m, dict) and m.get("role") == "user":
            said = m.get("content", "")
            if isinstance(said, list):
                said = " ".join(p.get("text", "") for p in said if isinstance(p, dict))
            return str(said).strip(), m
    return "", None


async def _act(params: FunctionCallParams, action: str, args: dict, confirm: str) -> None:
    """What every tool does: POST once per turn, tell the listeners, say the confirmation."""
    d = DISPATCH
    quiet = FunctionCallResultProperties(run_llm=False)      # one LLM pass per turn, no spoken LLM reply
    said, turn = last_user_turn(params.context)             # the turn this call answers, from its own context
    if turn is not None and turn is d.acted_on:             # the model called a second tool for the same turn
        await params.result_callback({"ok": False, "ignored": "one action per turn"}, properties=quiet)
        return
    d.acted_on = turn
    llm_ms = (time.monotonic() - d.turn_started) * 1000 if d.turn_started else None
    t0 = time.monotonic()
    delivered, answer = await d.post(action, args, said)
    post_ms = (time.monotonic() - t0) * 1000
    if isinstance(answer.get("say"), str):                   # the main app may word the confirmation itself
        confirm = answer["say"]
    if not delivered:
        confirm = ""                                         # never confirm what did not arrive
    sent = Sent(action, args, said, delivered, confirm, llm_ms, post_ms)
    for fn in list(d.listeners):
        fn(sent)
    await params.result_callback({"ok": delivered, "say": confirm}, properties=quiet)
    if confirm and d.speak:
        await params.pipeline_worker.queue_frame(TTSSpeakFrame(confirm, append_to_context=False))


# ---- teaching ----

async def start_teaching(params: FunctionCallParams, name: str = ""):
    """The person is about to demonstrate a task and wants the robot to watch and learn it.
    Examples: "watch me", "let me show you how to pack this", "I'll teach you the sorting".

    Args:
        name: A short name for the task if the person gave one ("sorting", "pack the box"), else empty.
    """
    await _act(params, "start_teaching", {"name": name}, "Watching.")


async def stop_teaching(params: FunctionCallParams):
    """The person has finished demonstrating. Only for words like "that's it", "I'm done showing you",
    "that's how you do it", "got that?". A bare "stop" is NOT this: use pause."""
    await _act(params, "stop_teaching", {}, "Got it.")


# ---- running ----

async def help_me(params: FunctionCallParams):
    """The person will do the task and wants the arm to assist: hand over the next item, one at a time.
    Examples: "help me", "give me a hand", "hand me the next one", "pass them to me"."""
    await _act(params, "help_me", {}, "Helping.")


async def your_turn(params: FunctionCallParams):
    """The arm should now do the whole task by itself. Examples: "your turn", "now you do it",
    "go ahead", "do it yourself", "show me what you learned"."""
    await _act(params, "your_turn", {}, "My turn.")


async def pause(params: FunctionCallParams):
    """Stop the arm moving right now. Use for any bare "stop", "wait", "hold on", "freeze", "pause",
    "careful", "whoa". When the person may want the arm to stop, choose this."""
    await _act(params, "pause", {}, "Paused.")


async def resume(params: FunctionCallParams):
    """Carry on after a pause. Examples: "continue", "carry on", "keep going", "okay go on", "resume"."""
    await _act(params, "resume", {}, "Continuing.")


async def load_task(params: FunctionCallParams, query: str):
    """Bring back a task that was taught earlier, named by the person by its name, its content or the
    time it was taught. Examples: "do what I taught you at ten", "the sorting one", "the task from this morning".

    Args:
        query: The person's own words for which task, e.g. "what I taught at 10", "the sorting one".
    """
    await _act(params, "load_task", {"query": query}, "Loading %s." % query if query else "Loading it.")


async def retry_last(params: FunctionCallParams):
    """Do the last step again because it went wrong. Examples: "try that again", "you dropped it, again",
    "redo that one", "you missed"."""
    await _act(params, "retry_last", {}, "Trying again.")


async def skip_step(params: FunctionCallParams):
    """Leave out the current or next step and move on. Examples: "skip that one", "never mind that one",
    "leave it, next"."""
    await _act(params, "skip_step", {}, "Skipping.")


# ---- corrections to the task ----

async def swap_steps(params: FunctionCallParams, a: str, b: str):
    """Exchange the places of two steps in the order. Examples: "swap the red and the green",
    "those two are the wrong way round", "switch the cup and the block".

    Args:
        a: One step, in the person's words: an item ("yellow block") or a position ("step 2", "the first one").
        b: The other step, named the same way.
    """
    await _act(params, "swap_steps", {"a": a, "b": b}, "Okay, swapped %s and %s." % (a, b))


async def move_step(params: FunctionCallParams, step: str, before: str):
    """Move one step earlier or later so that it happens just before another one. Examples: "the yellow
    one should go before the blue one", "do the cup first" (before = "first"), "red goes last" (before = "end").

    Args:
        step: The step to move, in the person's words: an item ("yellow one") or a position ("step 3").
        before: The step it must come before, named the same way; "first" for the very start, "end" for last.
    """
    await _act(params, "move_step", {"step": step, "before": before}, "Okay, %s before %s." % (step, before))


async def change_destination(params: FunctionCallParams, step_or_item: str, destination: str):
    """Change where an item is put. Examples: "the red one goes in the bowl, not the cup",
    "put that on the left instead", "no, blue ones belong in the box".

    Args:
        step_or_item: Which item or step, in the person's words ("red block", "that one", "step 2").
        destination: The new place or container, in the person's words ("bowl", "left of the cup").
    """
    await _act(params, "change_destination", {"step_or_item": step_or_item, "destination": destination},
               "Okay, %s to %s." % (step_or_item, destination))


async def set_grip(params: FunctionCallParams, level: Literal["firmer", "gentler"]):
    """Change how hard the gripper squeezes. "grip it harder", "hold it tighter", "it's slipping" are firmer;
    "you're crushing it", "softer", "be gentle with it" are gentler.

    Args:
        level: firmer or gentler.
    """
    await _act(params, "set_grip", {"level": level}, "Okay, %s grip." % level)


async def set_speed(params: FunctionCallParams, level: Literal["slower", "faster"]):
    """Change how fast the arm moves. "slow down", "not so fast", "take it easy" are slower;
    "speed up", "hurry up", "quicker" are faster.

    Args:
        level: slower or faster.
    """
    await _act(params, "set_speed", {"level": level}, "Okay, %s." % level)


# ---- everything else ----

async def nothing(params: FunctionCallParams):
    """The words are not an instruction to the robot arm: talk between people, a question to someone else,
    thinking aloud, a description of what the person is doing while demonstrating ("now the red one goes in
    the cup"), background speech, half a sentence, or anything unclear. This is the right choice for most
    of what is heard. When unsure, choose this."""
    await _act(params, "nothing", {}, "")


TOOLS = [start_teaching, stop_teaching, help_me, your_turn, pause, resume, load_task, swap_steps, move_step,
         change_destination, set_grip, set_speed, retry_last, skip_step, nothing]

SYSTEM_INSTRUCTION = """You are the ears of a robot arm on a table. A wristband microphone hears the whole room: \
the person teaching the arm, other people, and background talk. You get one piece of speech at a time, as text \
from a speech recogniser, so words may be slightly wrong.

Choose exactly one tool for each user turn. Never answer in text.
- Most speech is not for the robot. Then choose nothing().
- Choose an action only when the words clearly ask the arm to do something or change how it does the task.
- A bare "stop", "wait" or "hold on" is pause(), never stop_teaching().
- While the person demonstrates, they describe what they do ("the red one goes in the cup first"). \
That is narration, not a correction: choose nothing(). A correction disagrees with what the arm did or \
will do ("no, the red one should go before the blue one").
- Fill arguments with the person's own words for items, steps and places. Do not invent step numbers.
- Earlier turns are context only. Act on the newest user turn alone.
- When unsure, choose nothing().
If you ever must speak, keep it under eight words."""
