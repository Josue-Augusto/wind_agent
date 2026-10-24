from __future__ import annotations

import asyncio
import logging
import os
import tomllib
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

import httpx
import sounddevice as sd

from src.domain.entities.agent_state import AgentState
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.workers.runner import WorkerRunner


_LOGGER = logging.getLogger(__name__)
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_CONFIG_FILE = Path(__file__).resolve().parents[3] / "config.toml"


@dataclass(frozen=True)
class LocalPipecatSettings:
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_model: str | None = None
    whisper_model: str = "small"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    piper_voice: str = "pt_BR-faber-medium"
    audio_sample_rate: int = 16_000
    audio_channels: int = 1

    def __post_init__(self) -> None:
        parsed_url = urlsplit(self.lm_studio_base_url)
        if (
            parsed_url.scheme not in {"http", "https"}
            or parsed_url.hostname not in _LOCAL_HOSTS
        ):
            raise ValueError("LM Studio must use a localhost URL to keep inference local.")
        if self.audio_sample_rate <= 0:
            raise ValueError("audio_sample_rate must be greater than zero.")
        if self.audio_channels != 1:
            raise ValueError("The Whisper pipeline currently supports mono audio only.")
        if not self.whisper_model.strip():
            raise ValueError("whisper_model cannot be empty.")
        if not self.piper_voice.strip():
            raise ValueError("piper_voice cannot be empty.")
        if self.lm_studio_model is not None and not self.lm_studio_model.strip():
            raise ValueError("lm_studio_model cannot be empty when provided.")
        if not self.whisper_device.strip() or not self.whisper_compute_type.strip():
            raise ValueError("Whisper device and compute type cannot be empty.")

    @classmethod
    def from_environment(cls) -> "LocalPipecatSettings":
        try:
            with _CONFIG_FILE.open("rb") as config_file:
                config = tomllib.load(config_file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Required configuration file not found: {_CONFIG_FILE}"
            ) from error

        lm_studio_config = config.get("lm_studio")
        if not isinstance(lm_studio_config, dict):
            raise ValueError(
                f"Missing [lm_studio] section in configuration file: {_CONFIG_FILE}"
            )
        lm_studio_model = lm_studio_config.get("model")
        if not isinstance(lm_studio_model, str) or not lm_studio_model.strip():
            raise ValueError(
                f"Set a non-empty 'model' in [lm_studio] in {_CONFIG_FILE}"
            )

        return cls(
            lm_studio_base_url=os.getenv(
                "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
            ),
            lm_studio_model=lm_studio_model.strip(),
            whisper_model=os.getenv("WHISPER_MODEL", "small"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cuda"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
            piper_voice=os.getenv("PIPER_VOICE", "pt_BR-faber-medium"),
        )


class _SoundDeviceInputProcessor(FrameProcessor):
    def __init__(self, sample_rate: int, channels: int) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._channels = channels
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.InputStream | None = None
        self._active = False

    def start_capture(self) -> None:
        if self._stream is not None:
            raise RuntimeError("Microphone capture has already been started.")

        self._loop = asyncio.get_running_loop()
        stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="int16",
            blocksize=max(1, self._sample_rate // 50),
            callback=self._on_audio,
        )
        try:
            stream.start()
        except Exception:
            stream.close()
            raise
        self._stream = stream

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._active = True

    def _on_audio(self, audio, frame_count, time_info, status) -> None:
        if status:
            _LOGGER.warning("Microphone audio callback reported: %s", status)

        loop = self._loop
        if not self._active or loop is None or loop.is_closed():
            return

        frame = InputAudioRawFrame(
            audio=audio.tobytes(),
            sample_rate=self._sample_rate,
            num_channels=self._channels,
        )
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.push_frame(frame), loop
            )
        except RuntimeError:
            _LOGGER.exception("Could not enqueue microphone audio in the Pipecat loop.")
            return
        future.add_done_callback(self._log_frame_error)

    @staticmethod
    def _log_frame_error(future: Future[None]) -> None:
        try:
            future.result()
        except Exception:
            _LOGGER.exception("Could not process a microphone audio frame.")

    async def cleanup(self) -> None:
        self._active = False
        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                await asyncio.to_thread(self._close_stream, stream)
        finally:
            await asyncio.sleep(0)
            await super().cleanup()

    @staticmethod
    def _close_stream(stream: sd.InputStream) -> None:
        if stream.active:
            stream.stop()
        stream.close()


class _SoundDeviceOutputProcessor(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()
        self._stream: sd.RawOutputStream | None = None
        self._stream_format: tuple[int, int] | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM and isinstance(
            frame, TTSAudioRawFrame
        ):
            await asyncio.to_thread(self._write_audio, frame)
        await self.push_frame(frame, direction)

    def _write_audio(self, frame: TTSAudioRawFrame) -> None:
        audio_format = (frame.sample_rate, frame.num_channels)
        if self._stream is None or self._stream_format != audio_format:
            self._close_stream()
            self._stream = sd.RawOutputStream(
                samplerate=frame.sample_rate,
                channels=frame.num_channels,
                dtype="int16",
            )
            self._stream.start()
            self._stream_format = audio_format

        self._stream.write(frame.audio)

    async def cleanup(self) -> None:
        try:
            await asyncio.to_thread(self._close_stream)
        finally:
            await asyncio.sleep(0)
            await super().cleanup()

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        self._stream_format = None
        if stream is not None:
            if stream.active:
                stream.stop()
            stream.close()


class _AgentStateProcessor(FrameProcessor):
    def __init__(self, on_state_change: Callable[[AgentState], None]) -> None:
        super().__init__()
        self._on_state_change = on_state_change

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, (TranscriptionFrame, LLMFullResponseStartFrame)):
                self._on_state_change(AgentState.PROCESSING)
            elif isinstance(frame, TTSStartedFrame):
                self._on_state_change(AgentState.SPEAKING)
            elif isinstance(frame, (TTSStoppedFrame, LLMFullResponseEndFrame)):
                self._on_state_change(AgentState.LISTENING)
        await self.push_frame(frame, direction)


class LocalPipecatDataSource:
    def __init__(
        self,
        settings: LocalPipecatSettings | None = None,
        on_state_change: Callable[[AgentState], None] | None = None,
    ) -> None:
        self._settings = settings or LocalPipecatSettings.from_environment()
        self._on_state_change = on_state_change
        self._session_lock = asyncio.Lock()
        self._runner_task: asyncio.Task[None] | None = None
        self._runner: WorkerRunner | None = None
        self._pipeline_worker: PipelineWorker | None = None
        self._input_processor: _SoundDeviceInputProcessor | None = None

    def set_state_listener(
        self, listener: Callable[[AgentState], None] | None
    ) -> None:
        self._on_state_change = listener

    async def start_session(self) -> None:
        async with self._session_lock:
            if self._runner_task is not None and not self._runner_task.done():
                raise RuntimeError("A Pipecat voice session is already running.")
            await self._clear_finished_session()

            from pipecat.pipeline.pipeline import Pipeline
            from pipecat.pipeline.worker import PipelineParams, PipelineWorker
            from pipecat.processors.aggregators.llm_context import LLMContext
            from pipecat.processors.aggregators.llm_response_universal import (
                LLMContextAggregatorPair,
            )
            from pipecat.services.openai.llm import OpenAILLMService
            from pipecat.services.piper.tts import PiperTTSService
            from pipecat.services.whisper.stt import WhisperSTTService
            from pipecat.transcriptions.language import Language
            from pipecat.workers.runner import WorkerRunner

            model = await self._resolve_lm_studio_model()
            llm = OpenAILLMService(
                settings=OpenAILLMService.Settings(model=model),
                api_key="lm-studio",
                base_url=self._settings.lm_studio_base_url,
            )
            stt = WhisperSTTService(
                device=self._settings.whisper_device,
                compute_type=self._settings.whisper_compute_type,
                settings=WhisperSTTService.Settings(
                    model=self._settings.whisper_model,
                    language=Language.PT,
                ),
                sample_rate=self._settings.audio_sample_rate,
                num_channels=self._settings.audio_channels,
            )
            tts = PiperTTSService(
                settings=PiperTTSService.Settings(voice=self._settings.piper_voice)
            )

            context = LLMContext(
                [
                    {
                        "role": "system",
                        "content": (
                            "Você é um assistente de voz. Responda sempre em "
                            "português brasileiro, de forma clara e concisa."
                        ),
                    }
                ]
            )
            context_aggregator = LLMContextAggregatorPair(context)
            input_processor = _SoundDeviceInputProcessor(
                self._settings.audio_sample_rate, self._settings.audio_channels
            )
            output_processor = _SoundDeviceOutputProcessor()
            state_processor = _AgentStateProcessor(self._notify_state)
            try:
                pipeline = Pipeline(
                    [
                        input_processor,
                        stt,
                        context_aggregator.user(),
                        llm,
                        tts,
                        output_processor,
                        state_processor,
                        context_aggregator.assistant(),
                    ]
                )
                pipeline_worker = PipelineWorker(
                    pipeline,
                    params=PipelineParams(
                        audio_in_sample_rate=self._settings.audio_sample_rate,
                        audio_out_sample_rate=24_000,
                    ),
                    enable_rtvi=False,
                    idle_timeout_secs=None,
                )
                runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
                await runner.add_workers(pipeline_worker)
                input_processor.start_capture()
            except Exception:
                try:
                    await input_processor.cleanup()
                finally:
                    try:
                        await output_processor.cleanup()
                    finally:
                        await state_processor.cleanup()
                raise

            self._input_processor = input_processor
            self._pipeline_worker = pipeline_worker
            self._runner = runner
            self._runner_task = asyncio.create_task(runner.run())
            self._runner_task.add_done_callback(self._log_session_error)
            await asyncio.sleep(0)
            if self._runner_task.done():
                await self._runner_task
                raise RuntimeError("Pipecat worker stopped while starting the session.")
            self._notify_state(AgentState.LISTENING)

    async def stop_session(self) -> None:
        async with self._session_lock:
            runner_task = self._runner_task
            if runner_task is None:
                return

            try:
                if not runner_task.done() and self._runner is not None:
                    await self._runner.cancel()
                await runner_task
            finally:
                try:
                    await self._cleanup_input_processor()
                finally:
                    self._runner_task = None
                    self._runner = None
                    self._pipeline_worker = None
                    self._notify_state(AgentState.IDLE)

    async def _resolve_lm_studio_model(self) -> str:
        configured_model = self._settings.lm_studio_model
        if configured_model:
            return configured_model

        endpoint = f"{self._settings.lm_studio_base_url.rstrip('/')}/models"
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(endpoint)
            response.raise_for_status()

        payload = response.json()
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise RuntimeError(
                "LM Studio returned an invalid /v1/models response; "
                "set LM_STUDIO_MODEL to select a loaded model."
            )
        for model in models:
            if isinstance(model, dict) and isinstance(model.get("id"), str):
                model_id = model["id"].strip()
                if model_id:
                    return model_id

        raise RuntimeError(
            "No model is loaded in LM Studio. Load a model or set LM_STUDIO_MODEL."
        )

    async def _clear_finished_session(self) -> None:
        runner_task = self._runner_task
        if runner_task is not None and runner_task.done():
            try:
                await runner_task
            finally:
                try:
                    await self._cleanup_input_processor()
                finally:
                    self._runner_task = None
                    self._runner = None
                    self._pipeline_worker = None
                    self._notify_state(AgentState.IDLE)

    async def _cleanup_input_processor(self) -> None:
        input_processor, self._input_processor = self._input_processor, None
        if input_processor is not None:
            await input_processor.cleanup()

    def _notify_state(self, state: AgentState) -> None:
        if self._on_state_change is not None:
            self._on_state_change(state)

    def _log_session_error(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            _LOGGER.error(
                "Pipecat voice session failed.",
                exc_info=(type(error), error, error.__traceback__),
            )
        self._notify_state(AgentState.IDLE)
