"""The Pipecat pipeline of the voice loop.

    band PCM -> [count] -> Deepgram STT -> transcript log -> user turn (Silero VAD)
             -> turn gate -> LLM with the action tools -> [TTS -> speaker] -> assistant context

The LLM is forced to answer with a tool call and the tool result never starts a second
inference, so one spoken turn costs one LLM request. The short confirmation is spoken by
pushing the tool's own sentence to the TTS stage, not by asking the LLM for words.
"""
from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from typing import Optional

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, InputAudioRawFrame, LLMContextFrame, TTSAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from earshot import config
from earshot.voice import actions
from earshot.voice.band_input import SAMPLE_RATE, EchoGate, TranscriptLog

GOOGLE_MODEL = os.environ.get("EARSHOT_VOICE_MODEL", "gemini-3.5-flash-lite")
OPENAI_MODEL = os.environ.get("EARSHOT_VOICE_OPENAI_MODEL", "")       # empty: Pipecat's default model
ELEVENLABS_VOICE = os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
KEEP_TURNS = 3            # user turns kept as context; the band hears the room all day
ECHO_TAIL_S = 0.3         # the band stays deaf this long after the speaker went quiet


def get_key(name: str) -> Optional[str]:
    """An API key, or None when it is not set. Never printed."""
    try:
        return config.key(name) or None
    except RuntimeError:
        return None


class FrameCounter(FrameProcessor):
    """First stage: counts the band's PCM frames, so a run can show that sound arrived."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.frames = 0
        self.bytes = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self.frames += 1
            self.bytes += len(frame.audio)
        await self.push_frame(frame, direction)


class TurnGate(FrameProcessor):
    """Just before the LLM: notes when the request left (for the timing that is printed) and
    cuts the context down to the last few turns."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            messages = list(frame.context.get_messages())     # a copy: set_messages changes the list in place
            users = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "user"]
            if len(users) > KEEP_TURNS:
                frame.context.set_messages(messages[users[-KEEP_TURNS]:])
            actions.DISPATCH.begin_turn()
        await self.push_frame(frame, direction)


class SpeakerOutput(FrameProcessor):
    """Plays TTS audio on the laptop's default output with sounddevice (bundled PortAudio, so no
    PyAudio and nothing to brew). Output only: it never opens a microphone. The device is opened
    at the first audio frame. While it plays, the echo gate keeps the band's sound out."""

    def __init__(self, echo_gate: Optional[EchoGate] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.echo_gate = echo_gate
        self.played_frames = 0
        self._queue: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._rate = 0
        self._broken = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame) and frame.audio and not self._broken:
            if self._thread is None:
                self._rate = frame.sample_rate
                self._thread = threading.Thread(target=self._play, args=(frame.sample_rate, frame.num_channels),
                                                daemon=True)
                self._thread.start()
            if self.echo_gate is not None:
                if not self.echo_gate.active:             # a new utterance: add the tail once
                    self.echo_gate.hold(ECHO_TAIL_S)
                self.echo_gate.hold(len(frame.audio) / (2 * frame.num_channels * frame.sample_rate))
            self._queue.put(frame.audio)
            self.played_frames += 1
        await self.push_frame(frame, direction)

    def _play(self, rate: int, channels: int) -> None:
        try:
            import sounddevice
            with sounddevice.RawOutputStream(samplerate=rate, channels=channels, dtype="int16") as out:
                while True:
                    chunk = self._queue.get()
                    if chunk is None:
                        return
                    out.write(chunk)
        except Exception as e:                    # no output device, or it went away: carry on silently
            self._broken = True
            print("[voice] The speaker cannot be used (%s); carrying on without spoken replies."
                  % (str(e).splitlines()[0] if str(e) else type(e).__name__), flush=True)

    async def cleanup(self):
        await super().cleanup()
        if self._thread is not None:
            self._queue.put(None)                 # play what is queued, then close
            self._thread.join(timeout=10)
            self._thread = None


@dataclass
class Built:
    worker: PipelineWorker
    counter: FrameCounter
    transcripts: Optional[TranscriptLog]
    speaker: Optional[SpeakerOutput]
    echo_gate: EchoGate
    notes: list            # one sentence for each thing that is switched off, and why
    stt: bool
    llm: bool
    tts: str               # "elevenlabs", "deepgram" or ""


def _llm(kind: str, notes: list):
    if kind == "openai":
        api_key = get_key("OPENAI_API_KEY")
        if not api_key:
            notes.append("OPENAI_API_KEY is not set, so no action can be chosen with --llm openai; "
                         "put it in .env or use --llm google.")
            return None
        from pipecat.services.openai.llm import OpenAILLMService
        settings = {"system_instruction": actions.SYSTEM_INSTRUCTION, "temperature": 0}
        if OPENAI_MODEL:
            settings["model"] = OPENAI_MODEL
        return OpenAILLMService(api_key=api_key, settings=OpenAILLMService.Settings(**settings))
    api_key = get_key("GEMINI_API_KEY")
    if not api_key:
        notes.append("GEMINI_API_KEY is not set, so no action can be chosen; put it in .env.")
        return None
    from pipecat.services.google.llm import GoogleLLMService
    return GoogleLLMService(
        api_key=api_key,
        settings=GoogleLLMService.Settings(model=GOOGLE_MODEL, system_instruction=actions.SYSTEM_INSTRUCTION,
                                           temperature=0),
        tool_config={"function_calling_config": {"mode": "ANY"}},        # always a tool, never prose
    )


def _tts(notes: list):
    key = get_key("ELEVENLABS_API_KEY")
    if key:
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
        return "elevenlabs", ElevenLabsTTSService(api_key=key,
                                                  settings=ElevenLabsTTSService.Settings(voice=ELEVENLABS_VOICE))
    key = get_key("DEEPGRAM_API_KEY")
    if key:
        from pipecat.services.deepgram.tts import DeepgramTTSService
        return "deepgram", DeepgramTTSService(api_key=key)
    notes.append("Neither ELEVENLABS_API_KEY nor DEEPGRAM_API_KEY is set, so confirmations are printed, not spoken.")
    return "", None


def build(*, listen: bool, llm_kind: str = "google", tts: bool = True, post_transcripts: bool = True) -> Built:
    """Build the worker. listen=False is the typed-text form: no STT and no voice detection."""
    notes: list = []
    echo_gate = EchoGate()
    counter = FrameCounter()
    stages: list = [counter]

    transcripts = None
    has_stt = False
    if listen:
        dg = get_key("DEEPGRAM_API_KEY")
        if dg:
            from pipecat.services.deepgram.stt import DeepgramSTTService
            stages.append(DeepgramSTTService(
                api_key=dg, sample_rate=SAMPLE_RATE,
                settings=DeepgramSTTService.Settings(model="nova-3-general", smart_format=True, numerals=True)))
            transcripts = TranscriptLog(post=post_transcripts)
            stages.append(transcripts)
            has_stt = True
        else:
            notes.append("DEEPGRAM_API_KEY is not set, so the band's sound is received but not turned into text; "
                         "put it in .env, or test the actions with --say \"words\".")

    llm = _llm(llm_kind, notes)
    context = LLMContext(tools=list(actions.TOOLS), **({"tool_choice": "required"} if llm_kind == "openai" else {}))
    quiet = dict(enable_interruptions=False)       # room talk must not cancel an action that is being chosen
    user_params = LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer() if listen else None,
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy(**quiet), TranscriptionUserTurnStartStrategy(**quiet)],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.7)]),
        empty_user_turn=None,                      # sound without words never reaches the LLM
    )
    pair = LLMContextAggregatorPair(context, user_params=user_params)

    tts_kind, speaker = "", None
    if llm is not None:
        stages += [pair.user(), TurnGate(), llm]
        if tts:
            tts_kind, tts_service = _tts(notes)
            if tts_service is not None:
                speaker = SpeakerOutput(echo_gate)
                stages += [tts_service, speaker]
        stages.append(pair.assistant())
    actions.DISPATCH.speak = speaker is not None

    worker = PipelineWorker(
        Pipeline(stages),
        params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=24000),
        idle_timeout_secs=None,                    # a quiet room is not a reason to stop
        enable_rtvi=False,
    )
    return Built(worker, counter, transcripts, speaker, echo_gate, notes, has_stt, llm is not None, tts_kind)
