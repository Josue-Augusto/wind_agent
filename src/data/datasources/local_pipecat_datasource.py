from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import tomllib
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING
from urllib.parse import urlsplit

import httpx
import sounddevice as sd

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
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
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.workers.runner import WorkerRunner


_LOGGER = logging.getLogger(__name__)
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_CONFIG_FILE = Path(__file__).resolve().parents[3] / "config.toml"
_AUDIO_SETTINGS_FILE = Path(
    os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
) / "WindAgent" / "settings.json"


@dataclass(frozen=True)
class LocalPipecatSettings:
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_model: str | None = None
    whisper_model: str = "small"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    piper_voice: str = "pt_BR-faber-medium"
    audio_input_device: int | str | None = None
    audio_input_gain: float = 4.0
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
        if not math.isfinite(self.audio_input_gain) or not 0 < self.audio_input_gain <= 32:
            raise ValueError("audio_input_gain must be greater than zero and at most 32.")
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

        input_device = os.getenv("AUDIO_INPUT_DEVICE")
        if input_device is not None:
            input_device = int(input_device) if input_device.isdecimal() else input_device

        return cls(
            lm_studio_base_url=os.getenv(
                "LM_STUDIO_BASE_URL", "http://localhost:1234/v1"
            ),
            lm_studio_model=lm_studio_model.strip(),
            whisper_model=os.getenv("WHISPER_MODEL", "small"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cuda"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
            piper_voice=os.getenv("PIPER_VOICE", "pt_BR-faber-medium"),
            audio_input_device=input_device,
            audio_input_gain=float(os.getenv("AUDIO_INPUT_GAIN", "4.0")),
        )


class _SoundDeviceInputProcessor(FrameProcessor):
    def __init__(
        self,
        sample_rate: int,
        channels: int,
        device: int | str | None = None,
        capture_channels: int = 1,
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._channels = channels
        self._device = device
        self._capture_channels = capture_channels
        self._gain = gain
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.InputStream | None = None
        self._active = False
        self._first_audio_frame_logged = False
        self._last_audio_level_logged_at = time.monotonic()
        self._audio_level_sample_count = 0
        self._audio_level_squared_sum = 0.0
        self._audio_level_peak = 0

    def start_capture(self) -> None:
        if self._stream is not None:
            raise RuntimeError("Microphone capture has already been started.")

        self._loop = asyncio.get_running_loop()
        selected_device = sd.query_devices(self._device, "input")
        host_api = sd.query_hostapis()[selected_device["hostapi"]]["name"]
        _LOGGER.info(
            "Selected audio input device: %s (%s), software gain up to %.1fx",
            selected_device["name"],
            host_api,
            self._gain,
        )
        stream = sd.InputStream(
            device=self._device,
            samplerate=self._sample_rate,
            channels=self._capture_channels,
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
        _LOGGER.info(
            "Microphone capture started on input device '%s'.", selected_device["name"]
        )

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

        if not self._first_audio_frame_logged:
            self._first_audio_frame_logged = True
            _LOGGER.info("Microphone audio frames are reaching the Pipecat pipeline.")

        if audio.ndim > 1:
            channel_rms = (audio.astype("float64") ** 2).mean(axis=0)
            audio = audio[:, int(channel_rms.argmax())]
        if self._gain != 1.0:
            audio = (
                (audio.astype("float32") * self._gain)
                .clip(-32768, 32767)
                .astype("int16")
            )

        audio_float = audio.astype("float32")
        self._audio_level_sample_count += audio.size
        self._audio_level_squared_sum += float((audio_float**2).sum())
        self._audio_level_peak = max(
            self._audio_level_peak,
            int(abs(audio.astype("int32")).max()) if audio.size else 0,
        )
        now = time.monotonic()
        if now - self._last_audio_level_logged_at >= 3.0:
            self._last_audio_level_logged_at = now
            peak = self._audio_level_peak
            rms = (
                (self._audio_level_squared_sum / self._audio_level_sample_count) ** 0.5
                / 32768.0
                if self._audio_level_sample_count
                else 0.0
            )
            self._audio_level_sample_count = 0
            self._audio_level_squared_sum = 0.0
            self._audio_level_peak = 0
            if peak <= 1 or rms < 0.001:
                _LOGGER.warning(
                    "Microphone input is silent or very quiet: peak=%d/32768, rms=%.4f. "
                    "Check Windows microphone permissions or set AUDIO_INPUT_DEVICE.",
                    peak,
                    rms,
                )
            else:
                _LOGGER.info(
                    "Microphone input level: peak=%d/32768, rms=%.4f.", peak, rms
                )

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
            if isinstance(frame, VADUserStartedSpeakingFrame):
                _LOGGER.info("VAD detected speech; waiting for the end of the utterance.")
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                _LOGGER.info("VAD detected the end of the utterance; Whisper will transcribe it.")
            elif isinstance(frame, LLMFullResponseStartFrame):
                _LOGGER.info("LM Studio started generating a response.")
                self._on_state_change(AgentState.PROCESSING)
            elif isinstance(frame, TTSStartedFrame):
                self._on_state_change(AgentState.SPEAKING)
            elif isinstance(frame, (TTSStoppedFrame, LLMFullResponseEndFrame)):
                self._on_state_change(AgentState.LISTENING)
        await self.push_frame(frame, direction)


class _TranscriptionLoggingProcessor(FrameProcessor):
    def __init__(self, on_state_change: Callable[[AgentState], None]) -> None:
        super().__init__()
        self._on_state_change = on_state_change

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM and isinstance(
            frame, TranscriptionFrame
        ):
            _LOGGER.info(
                "Whisper produced a transcription (%d characters); forwarding it to the conversation.",
                len(frame.text.strip()),
            )
            self._on_state_change(AgentState.PROCESSING)
        await self.push_frame(frame, direction)


class _DiagnosticSileroVADAnalyzer(SileroVADAnalyzer):
    def __init__(self, *, sample_rate: int, params: VADParams) -> None:
        super().__init__(sample_rate=sample_rate, params=params)
        self._last_confidence = 0.0
        self._last_diagnostic_at = time.monotonic()

    def voice_confidence(self, buffer: bytes) -> float:
        self._last_confidence = super().voice_confidence(buffer)
        return self._last_confidence

    def _get_smoothed_volume(self, audio: bytes) -> float:
        volume = super()._get_smoothed_volume(audio)
        now = time.monotonic()
        if now - self._last_diagnostic_at >= 1.0:
            self._last_diagnostic_at = now
            _LOGGER.info(
                "VAD diagnostic: confidence=%.3f (threshold=%.3f), "
                "volume=%.3f (threshold=%.3f).",
                self._last_confidence,
                self.params.confidence,
                volume,
                self.params.min_volume,
            )
        return volume


class LocalPipecatDataSource:
    def __init__(
        self,
        settings: LocalPipecatSettings | None = None,
        on_state_change: Callable[[AgentState], None] | None = None,
    ) -> None:
        self._settings = settings or LocalPipecatSettings.from_environment()
        self._audio_devices = self._discover_audio_input_devices()
        self._audio_input_device = self._resolve_initial_audio_input_device()
        self._on_state_change = on_state_change
        self._session_lock = asyncio.Lock()
        self._runner_task: asyncio.Task[None] | None = None
        self._runner: WorkerRunner | None = None
        self._pipeline_worker: PipelineWorker | None = None
        self._input_processor: _SoundDeviceInputProcessor | None = None

    def _discover_audio_input_devices(self) -> list[dict[str, object]]:
        host_apis = sd.query_hostapis()
        devices = sd.query_devices()
        candidates = [
            (index, device, host_apis[device["hostapi"]]["name"])
            for index, device in enumerate(devices)
            if device["max_input_channels"] >= self._settings.audio_channels
            and str(device["name"]).strip()
            and not any(
                marker in str(device["name"]).casefold()
                for marker in (
                    "mapper",
                    "mapeador",
                    "driver de captura de som primário",
                    "primary sound driver",
                )
            )
        ]
        unique_devices: list[dict[str, object]] = []
        grouped_devices: dict[str, list[tuple[int, Any, str]]] = {}
        for candidate in candidates:
            normalized_name = str(candidate[1]["name"]).strip().casefold()
            grouped_devices.setdefault(normalized_name, []).append(candidate)

        api_priority = {
            "MME": 0,
            "Windows DirectSound": 1,
            "Windows WASAPI": 2,
            "Windows WDM-KS": 3,
        }
        for name_candidates in grouped_devices.values():
            compatible_candidates = []
            for candidate in name_candidates:
                index, device, host_api_name = candidate
                try:
                    sd.check_input_settings(
                        device=index,
                        channels=self._settings.audio_channels,
                        dtype="int16",
                        samplerate=self._settings.audio_sample_rate,
                    )
                except sd.PortAudioError:
                    continue
                capture_channels = 1
                if device["max_input_channels"] >= 2:
                    try:
                        sd.check_input_settings(
                            device=index,
                            channels=2,
                            dtype="int16",
                            samplerate=self._settings.audio_sample_rate,
                        )
                        capture_channels = 2
                    except sd.PortAudioError:
                        pass
                compatible_candidates.append(
                    (*candidate, capture_channels)
                )
            if not compatible_candidates:
                continue
            index, device, host_api_name, capture_channels = min(
                compatible_candidates,
                key=lambda candidate: api_priority.get(candidate[2], 99),
            )
            unique_devices.append(
                {
                    "index": index,
                    "name": str(device["name"]).strip(),
                    "host_api": host_api_name,
                    "capture_channels": capture_channels,
                }
            )
        return unique_devices

    def get_audio_input_devices(self) -> list[tuple[int, str]]:
        return [
            (int(device["index"]), str(device["name"]))
            for device in self._audio_devices
        ]

    def _resolve_initial_audio_input_device(self) -> int:
        requested = self._settings.audio_input_device
        if requested is not None:
            try:
                raw_device = sd.query_devices(requested, "input")
                match = self._find_audio_device(str(raw_device["name"]))
                if match is not None:
                    return match
            except (ValueError, sd.PortAudioError):
                _LOGGER.warning(
                    "Configured audio input device %r is unavailable.", requested
                )

        saved_device = self._load_saved_audio_device()
        if saved_device is not None:
            match = next(
                (
                    int(device["index"])
                    for device in self._audio_devices
                    if device["name"] == saved_device.get("name")
                ),
                None,
            )
            if match is not None:
                _LOGGER.info(
                    "Restored saved audio input device: %s", saved_device["name"]
                )
                return match

        try:
            default_device = sd.query_devices(kind="input")
            match = self._find_audio_device(str(default_device["name"]))
            if match is not None:
                _LOGGER.info(
                    "Using the Windows default audio input: %s",
                    default_device["name"],
                )
                return match
        except (ValueError, sd.PortAudioError):
            pass

        if not self._audio_devices:
            raise RuntimeError("No active audio input devices were found.")
        fallback = int(self._audio_devices[0]["index"])
        _LOGGER.warning(
            "Windows default input was unavailable; using %s.",
            self._audio_devices[0]["name"],
        )
        return fallback

    def _find_audio_device(self, name: str) -> int | None:
        for device in self._audio_devices:
            if str(device["name"]).casefold() == name.casefold():
                return int(device["index"])
        return None

    @staticmethod
    def _load_saved_audio_device() -> dict[str, str] | None:
        try:
            settings = json.loads(_AUDIO_SETTINGS_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        audio_device = (
            settings.get("audio_input_device") if isinstance(settings, dict) else None
        )
        if (
            isinstance(audio_device, dict)
            and isinstance(audio_device.get("name"), str)
            and isinstance(audio_device.get("host_api"), str)
        ):
            return audio_device
        return None

    def get_selected_audio_input_device(self) -> int:
        return self._audio_input_device

    def get_audio_input_capture_channels(self) -> int:
        device = next(
            item
            for item in self._audio_devices
            if item["index"] == self._audio_input_device
        )
        return int(device["capture_channels"])

    def select_audio_input_device(self, device_index: int) -> None:
        selected_device = next(
            (device for device in self._audio_devices if device["index"] == device_index),
            None,
        )
        if selected_device is None:
            raise ValueError(f"Audio input device {device_index} is unavailable.")
        self._audio_input_device = device_index
        settings = {
            "audio_input_device": {
                "name": selected_device["name"],
                "host_api": selected_device["host_api"],
            }
        }
        try:
            _AUDIO_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            temporary_file = _AUDIO_SETTINGS_FILE.with_suffix(".tmp")
            temporary_file.write_text(
                json.dumps(settings, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_file.replace(_AUDIO_SETTINGS_FILE)
        except OSError:
            _LOGGER.exception("Could not save the selected audio input device.")

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
                self._settings.audio_sample_rate,
                self._settings.audio_channels,
                self._audio_input_device,
                self.get_audio_input_capture_channels(),
                self._settings.audio_input_gain,
            )
            vad_processor = VADProcessor(
                vad_analyzer=_DiagnosticSileroVADAnalyzer(
                    sample_rate=self._settings.audio_sample_rate,
                    params=VADParams(min_volume=0.2, stop_secs=0.8),
                )
            )
            output_processor = _SoundDeviceOutputProcessor()
            transcription_processor = _TranscriptionLoggingProcessor(self._notify_state)
            state_processor = _AgentStateProcessor(self._notify_state)
            try:
                pipeline = Pipeline(
                    [
                        input_processor,
                        vad_processor,
                        stt,
                        transcription_processor,
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
                        try:
                            await transcription_processor.cleanup()
                        finally:
                            try:
                                await state_processor.cleanup()
                            finally:
                                await vad_processor.cleanup()
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
                await self._cleanup_input_processor()
                pipeline_worker = self._pipeline_worker
                if not runner_task.done() and pipeline_worker is not None:
                    await pipeline_worker.end(reason="Voice session stopped by user")
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
