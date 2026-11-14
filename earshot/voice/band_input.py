"""The wristband as the pipeline's microphone, and the record of what it heard.

BandInput connects to one Vexo Open band (the team's number, or the kit's pretend band),
decodes its Opus packets to 16 kHz PCM and queues them into the Pipecat worker. Taps are
forwarded to the main app as {"action": "tap" | "double_tap"}. When the link drops the
library reconnects by itself; when it gives up, this keeps trying the SAME band.

TranscriptLog is the pipeline stage right after speech-to-text: every final transcript line
is printed, appended to data/transcripts/YYYYMMDD.jsonl with a wall-clock time, and POSTed
to the main app as {"action": "transcript", "args": {"text", "t"}} so the teaching step can
use the words spoken during a demonstration.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

from pipecat.frames.frames import Frame, InputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from earshot import config
from earshot.voice.actions import DISPATCH

KIT = config.ROOT / "vexo-open-kit"
SAMPLE_SPEECH = KIT / "dashboard" / "speech" / "sample-speech.opus"
TRANSCRIPTS_DIR = config.DATA / "transcripts"
SAMPLE_RATE = 16000
RETRY_S = 5.0


def _kit():
    """Import the band library from the cloned kit (it is used from its folder, not installed)."""
    if not (KIT / "python" / "vexo_open").is_dir():
        raise RuntimeError("The band kit is missing. Clone it once: git clone "
                           "https://github.com/VexoAI-Org/vexo-open-kit.git %s" % KIT)
    path = str(KIT / "python")
    if path not in sys.path:
        sys.path.insert(0, path)
    import vexo_open
    return vexo_open


class EchoGate:
    """Lets the speaker tell the band input "I am talking": the band's microphone hears the
    laptop speaker, so its sound is replaced by silence until the speech has ended."""

    def __init__(self) -> None:
        self._until = 0.0

    def hold(self, seconds: float) -> None:
        self._until = max(self._until, time.monotonic()) + seconds

    @property
    def active(self) -> bool:
        return time.monotonic() < self._until


class BandInput:
    def __init__(self, worker, number: Optional[int], transport: str = "ble",
                 echo_gate: Optional[EchoGate] = None, taps: bool = True, port: Optional[str] = None) -> None:
        if transport != "sim" and number is None:
            raise RuntimeError("A real band needs its number: --band N (the number on the band's bag).")
        self.worker = worker
        self.number = number              # None only for the pretend band
        self.transport = transport
        self.port = port                  # serial only: which USB port is the band (the arm is one too)
        self.echo_gate = echo_gate
        self.taps = taps
        self.band = None
        self.packets = 0                  # Opus packets received from the band
        self.frames = 0                   # PCM frames queued into the pipeline
        self.silenced = 0                 # of those, replaced by silence while the speaker talked
        self._decoder = None
        self._pcm: asyncio.Queue = asyncio.Queue(maxsize=500)    # 10 s of sound at most
        self._lost = asyncio.Event()
        self._closing = False

    # ---- band events ----

    def _on_audio(self, a) -> None:
        self.packets += 1
        try:
            pcm = self._decoder.feed(a)
        except Exception as e:                                  # one bad packet must not end the stream
            if self.packets % 500 == 1:
                print("[band] Could not decode a sound packet (%s)." % type(e).__name__, flush=True)
            return
        if not pcm:
            return
        if self.echo_gate is not None and self.echo_gate.active:
            pcm = bytes(len(pcm))
            self.silenced += 1
        try:
            self._pcm.put_nowait(pcm)
        except asyncio.QueueFull:
            pass                                                # the pipeline is stuck: drop, do not grow

    async def _on_tap(self, e) -> None:
        action = "double_tap" if e.kind == "double-tap" else "tap"
        print("[band] %s" % action, flush=True)
        await DISPATCH.post(action, {}, "")

    def _on_reconnecting(self, e) -> None:
        print("[band] Link lost; trying to get it back for %g seconds." % e.seconds, flush=True)

    def _on_reconnected(self, e) -> None:
        self._decoder = self._new_decoder()                     # the stream started again
        print("[band] Link is back.", flush=True)

    def _on_disconnect(self, e) -> None:
        if not self._closing:
            print("[band] Disconnected (%s)." % getattr(e, "reason", "?"), flush=True)
            self._lost.set()

    def _new_decoder(self):
        from vexo_open.audio import PcmDecoder
        return PcmDecoder()

    # ---- life cycle ----

    async def _connect(self) -> None:
        vexo = _kit()
        options = {}
        if self.transport == "sim":
            from vexo_open import audio
            options["audio_packets"] = audio.ogg_packets(str(SAMPLE_SPEECH))
        if self.transport == "serial" and self.port:
            options["port"] = self.port
        self._decoder = self._new_decoder()                     # slow the first time: before the mic goes on
        band = await vexo.Band.connect(self.number, transport=self.transport, auto_reconnect=True, **options)
        if self.number is not None and band.number != self.number:
            await band.disconnect()
            raise RuntimeError("Band %s answered, not band %s. Not using it." % (band.number, self.number))
        self.band = band
        band.on("audio", self._on_audio)
        band.on("reconnecting", self._on_reconnecting)
        band.on("reconnected", self._on_reconnected)
        band.on("disconnect", self._on_disconnect)
        if self.taps:
            band.on("tap", self._on_tap)
            band.on("double-tap", self._on_tap)
        await band.configure(audio=True, events=self.taps)      # the band's light is now solid blue
        print("[band] Listening to %s%s. Its microphone hears everyone nearby."
              % (band.name, " (pretend band, playing the kit's sample speech)" if self.transport == "sim" else ""),
              flush=True)

    async def _pump(self) -> None:
        while True:
            pcm = await self._pcm.get()
            await self.worker.queue_frame(InputAudioRawFrame(audio=pcm, sample_rate=SAMPLE_RATE, num_channels=1))
            self.frames += 1

    async def run(self) -> None:
        """Connect and feed the pipeline until cancelled. Reconnects to the same band for ever."""
        pump = asyncio.ensure_future(self._pump())
        try:
            while True:
                self._lost.clear()
                try:
                    await self._connect()
                except Exception as e:
                    text = str(e) or type(e).__name__
                    print("[band] Not connected: %s" % text.splitlines()[0], flush=True)
                    if self.transport == "sim":
                        raise RuntimeError("The pretend band could not start.") from None
                    print("[band] Double-tap the band (it can then be found for 60 seconds). Trying again in "
                          "%g seconds." % RETRY_S, flush=True)
                    await asyncio.sleep(RETRY_S)
                    continue
                await self._lost.wait()
                await self._close_band()
                await asyncio.sleep(RETRY_S)
        finally:
            pump.cancel()
            await self._close_band()

    async def _close_band(self) -> None:
        band, self.band = self.band, None
        if band is not None:
            self._closing = True
            try:
                await band.disconnect()
            except Exception:
                pass
            self._closing = False


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
