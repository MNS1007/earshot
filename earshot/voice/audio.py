"""Sound into the voice pipeline, from any microphone, and the record of what was heard.

MicInput reads a microphone the computer already knows (built-in, USB, a headset, a lapel
mic's receiver) and queues 16 kHz mono PCM into the Pipecat worker. It is the default
source. A wearable with its own radio link is just another source with the same shape:
see band_input.py for one.

TranscriptLog is the pipeline stage right after speech-to-text: every final transcript line
is printed, appended to data/transcripts/YYYYMMDD.jsonl with a wall-clock time, and POSTed
to the main app as {"action": "transcript", "args": {"text", "t"}} so the teaching step can
use the words spoken during a demonstration.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Optional, Union

from pipecat.frames.frames import Frame, InputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from earshot import config
from earshot.voice.actions import DISPATCH

TRANSCRIPTS_DIR = config.DATA / "transcripts"
SAMPLE_RATE = 16000
BLOCK_S = 0.02            # sound is handed over in 20 ms pieces


class EchoGate:
    """Lets the speaker tell the microphone "I am talking": the microphone hears the laptop
    speaker, so its sound is replaced by silence until the speech has ended."""

    def __init__(self) -> None:
        self._until = 0.0

    def hold(self, seconds: float) -> None:
        self._until = max(self._until, time.monotonic()) + seconds

    @property
    def active(self) -> bool:
        return time.monotonic() < self._until


def list_microphones() -> list[tuple[int, str, bool]]:
    """Every sound input the computer offers: (index, name, is the default)."""
    import sounddevice as sd
    try:
        default = sd.default.device[0]
    except Exception:
        default = -1
    return [(i, d["name"], i == default) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


class MicInput:
    """One microphone as the pipeline's ears. `device` is None for the system default, an
    index from list_microphones(), or part of a device's name."""

    def __init__(self, worker, device: Union[int, str, None] = None, echo_gate: Optional[EchoGate] = None) -> None:
        self.worker = worker
        self.device = device
        self.echo_gate = echo_gate
        self.packets = 0                  # blocks received from the microphone
        self.frames = 0                   # PCM frames queued into the pipeline
        self.silenced = 0                 # of those, replaced by silence while the speaker talked
        self._pcm: asyncio.Queue = asyncio.Queue(maxsize=500)    # 10 s of sound at most
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._rate = SAMPLE_RATE

    def _put(self, pcm: bytes) -> None:
        try:
            self._pcm.put_nowait(pcm)
        except asyncio.QueueFull:
            pass                                                # the pipeline is stuck: drop, do not grow

    def _on_block(self, indata, frames, when, status) -> None:
        """Called on the sound driver's own thread."""
        self.packets += 1
        pcm = bytes(indata)
        if self._rate != SAMPLE_RATE:
            pcm = self._to_16k(pcm)
        if self.echo_gate is not None and self.echo_gate.active:
            pcm = bytes(len(pcm))
            self.silenced += 1
        self._loop.call_soon_threadsafe(self._put, pcm)

    def _to_16k(self, pcm: bytes) -> bytes:
        """For a microphone that will not open at 16 kHz: straight-line resampling of each block."""
        import numpy as np
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        n = max(1, int(round(len(x) * SAMPLE_RATE / self._rate)))
        return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.int16).tobytes()

    def _open(self):
        import sounddevice as sd
        info = sd.query_devices(self.device, "input")
        for rate in (SAMPLE_RATE, int(info["default_samplerate"])):
            try:
                stream = sd.RawInputStream(samplerate=rate, channels=1, dtype="int16", device=self.device,
                                           blocksize=int(rate * BLOCK_S), callback=self._on_block)
            except Exception:
                if rate != SAMPLE_RATE:
                    raise
                continue
            self._rate = rate
            return stream, info["name"]

    async def _pump(self) -> None:
        while True:
            pcm = await self._pcm.get()
            await self.worker.queue_frame(InputAudioRawFrame(audio=pcm, sample_rate=SAMPLE_RATE, num_channels=1))
            self.frames += 1

    async def run(self) -> None:
        """Open the microphone and feed the pipeline until cancelled."""
        self._loop = asyncio.get_running_loop()
        try:
            stream, name = self._open()
        except Exception as e:
            raise RuntimeError("Could not open the microphone (%s). See the choices with --list-mics."
                               % (str(e).splitlines()[0] if str(e) else type(e).__name__)) from None
        pump = asyncio.ensure_future(self._pump())
        try:
            with stream:
                print('[mic] Listening to "%s". It hears everyone nearby, so tell the people around you.' % name,
                      flush=True)
                await asyncio.Event().wait()
        finally:
            pump.cancel()


class TranscriptLog(FrameProcessor):
    """Prints, saves and forwards every final transcript line; passes every frame on unchanged."""

    def __init__(self, post: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.post = post
        self.lines = 0
        self._tasks: set = set()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._record(frame.text.strip())
        await self.push_frame(frame, direction)

    def _record(self, text: str) -> None:
        t = time.time()
        self.lines += 1
        print('heard: "%s"' % text, flush=True)
        try:
            TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(TRANSCRIPTS_DIR / (time.strftime("%Y%m%d", time.localtime(t)) + ".jsonl"), "a") as f:
                f.write(json.dumps({"t": t, "clock": time.strftime("%H:%M:%S", time.localtime(t)), "text": text})
                        + "\n")
        except OSError as e:
            print("[voice] Could not save the transcript line (%s)." % e, flush=True)
        if self.post:                                           # not awaited: a slow app must not delay the turn
            task = asyncio.ensure_future(DISPATCH.post("transcript", {"text": text, "t": t}, text))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
