"""The voice loop of Earshot. Run it with the voice virtualenv's Python:

    .venv-voice/bin/python scripts/voice_run.py --band N          the team's band, by its number
    .venv-voice/bin/python scripts/voice_run.py --sim             the kit's pretend band (sample speech)
    .venv-voice/bin/python scripts/voice_run.py --say "your turn" typed words, no band, no speech-to-text

--say may be given several times. --seconds S ends a band run after S seconds. --no-tts
keeps it silent. --llm google|openai picks the model that chooses the action.
The band number can also come from the environment variable EARSHOT_BAND.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Earshot voice loop: band -> text -> one action -> main app")
    p.add_argument("--band", type=int, default=None, help="the number on your band's bag (or set EARSHOT_BAND)")
    p.add_argument("--sim", action="store_true", help="use the kit's pretend band, which plays its sample speech")
    p.add_argument("--say", action="append", default=[], metavar="WORDS",
                   help="typed words instead of the band; skips speech-to-text (may be repeated)")
    p.add_argument("--no-tts", action="store_true", help="never speak, only print")
    p.add_argument("--llm", choices=["google", "openai"], default="google")
    p.add_argument("--seconds", type=float, default=None, help="stop a band run after this many seconds")
    p.add_argument("--no-taps", action="store_true", help="do not forward the band's taps")
    p.add_argument("--debug", action="store_true", help="show Pipecat's own log lines")
    args = p.parse_args()
    if args.band is None and os.environ.get("EARSHOT_BAND", "").strip().isdigit():
        args.band = int(os.environ["EARSHOT_BAND"])
    if not args.say and not args.sim and args.band is None:
        p.error('say which band: --band N (the number on its bag), or --sim for the pretend band, '
                'or --say "words" to type instead')
    return args


def show(sent) -> None:
    timing = "" if sent.llm_ms is None else "  [chosen in %.0f ms, POST %.0f ms]" % (sent.llm_ms, sent.post_ms)
    print("action: %s %s  %s%s" % (sent.action, sent.args or "", "sent" if sent.delivered else "NOT delivered",
                                   timing), flush=True)
    if sent.confirm:
        print('reply: "%s"' % sent.confirm, flush=True)


async def say_mode(args, built) -> int:
    from pipecat.frames.frames import LLMMessagesAppendFrame
    from pipecat.workers.runner import WorkerRunner
    from earshot.voice.actions import DISPATCH

    got: asyncio.Queue = asyncio.Queue()
    DISPATCH.listeners.append(got.put_nowait)
    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(built.worker)
    running = asyncio.ensure_future(runner.run())
    missed = 0
    for words in args.say:
        print('typed: "%s"' % words, flush=True)
        t0 = time.monotonic()
        await built.worker.queue_frame(LLMMessagesAppendFrame(messages=[{"role": "user", "content": words}],
                                                              run_llm=True))
        try:
            sent = await asyncio.wait_for(got.get(), timeout=20)
            show(sent)
            print("        typed text to POST finished: %.0f ms" % ((time.monotonic() - t0) * 1000), flush=True)
        except asyncio.TimeoutError:
            missed += 1
            print("action: none (the model did not answer with an action within 20 seconds)", flush=True)
        if running.done():
            break
    await built.worker.stop_when_done()
    await running
    return 1 if missed else 0


async def band_mode(args, built) -> int:
    from pipecat.workers.runner import WorkerRunner
    from earshot.voice.actions import DISPATCH
    from earshot.voice.band_input import BandInput

    DISPATCH.listeners.append(show)
    band = BandInput(built.worker, None if args.sim and args.band is None else args.band,
                     "sim" if args.sim else "ble", echo_gate=built.echo_gate, taps=not args.no_taps)
    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(built.worker)
    running = asyncio.ensure_future(runner.run())
    feeding = asyncio.ensure_future(band.run())
    timer = asyncio.ensure_future(asyncio.sleep(args.seconds)) if args.seconds else None
    waiting = {running, feeding} | ({timer} if timer else set())
    await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
    code = 0
    if feeding.done() and not feeding.cancelled() and feeding.exception() is not None:
        print("Stopped: %s" % feeding.exception(), flush=True)
        code = 1
    feeding.cancel()
    await asyncio.gather(feeding, return_exceptions=True)
    if timer:
        timer.cancel()
    if not running.done():
        await built.worker.cancel()
    await asyncio.gather(running, return_exceptions=True)
    print("Band packets received: %d. PCM frames that reached the pipeline: %d (%.1f s of sound). "
          "Final transcript lines: %s." % (band.packets, built.counter.frames, built.counter.bytes / 32000,
                                           built.transcripts.lines if built.transcripts else "none (no speech-to-text)"),
          flush=True)
    return code


async def main() -> int:
    args = parse()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.debug else "ERROR")
    from earshot.voice import pipeline
    from earshot.voice.actions import DISPATCH, post_url

    listen = not args.say
    built = pipeline.build(listen=listen, llm_kind=args.llm, tts=not args.no_tts)
    for note in built.notes:
        print("Note: " + note, flush=True)
    if not built.llm and args.say:
        return 1
    print("Actions go to %s. Model: %s. Speech-to-text: %s. Spoken replies: %s."
          % (post_url(), (pipeline.GOOGLE_MODEL if args.llm == "google" else pipeline.OPENAI_MODEL or "OpenAI default")
             if built.llm else "none", "Deepgram" if built.stt else "off", built.tts or "off"), flush=True)
    try:
        return await (say_mode(args, built) if args.say else band_mode(args, built))
    finally:
        await DISPATCH.close()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
