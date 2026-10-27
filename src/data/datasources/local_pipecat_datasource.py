from __future__ import annotations

import asyncio
import ctypes
import hashlib
import json
import logging
import math
import os
import sys
import time
import tomllib
from collections.abc import AsyncGenerator, Callable, Iterable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass, replace
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
    ErrorFrame,
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
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.tracing.service_decorators import traced_tts

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.workers.runner import WorkerRunner


_LOGGER = logging.getLogger(__name__)
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_LLM_PROVIDER_LABELS = {
    "lm_studio": "LM Studio",
    "ollama": "Ollama",
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
}
_LLM_PROVIDER_URLS = {
    "lm_studio": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
}
_TTS_ENGINES = {
    "qwen3": "Qwen3-TTS local",
    "kokoro": "Kokoro-82M",
    "piper": "Piper",
}
_QWEN3_SPEED = {"slow": 0.88, "normal": 1.0, "fast": 1.12}
_SESSION_DRAIN_TIMEOUT_SECS = 60.0
_CONFIG_FILE = Path(__file__).resolve().parents[3] / "config.toml"
_USER_SETTINGS_FILE = Path(
    os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
) / "WindAgent" / "settings.json"
_KOKORO_MODEL_DIRECTORY = Path(
    os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
) / "WindAgent" / "kokoro"
_KOKORO_ASSETS = {
    "kokoro-v1.0.onnx": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.1/kokoro-v1.0.onnx"
    ),
    "voices-v1.0.bin": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
        "model-files-v1.1/voices-v1.0.bin"
    ),
}
_CONVERSATION_LANGUAGES = {
    "pt": ("Português (Brasil)", "pt_BR-faber-medium"),
    "en": ("English (US)", "en_US-lessac-medium"),
    "es": ("Español", "es_ES-davefx-medium"),
}
_PIPER_VOICES = {
    "pt": (
        ("pt_BR-faber-medium", "Faber (média)"),
        ("pt_BR-cadu-medium", "Cadu (média)"),
        ("pt_BR-edresson-low", "Edresson (leve)"),
    ),
    "en": (
        ("en_US-lessac-medium", "Lessac (média)"),
        ("en_US-amy-medium", "Amy (média)"),
        ("en_US-ryan-medium", "Ryan (média)"),
    ),
    "es": (
        ("es_ES-davefx-medium", "Davefx (média)"),
        ("es_MX-ald-medium", "Ald (média, México)"),
    ),
}
_KOKORO_VOICES = {
    "pt": (("pf_dora", "Dora"), ("pm_alex", "Alex"), ("pm_santa", "Papai Noel")),
    "en": (
        ("af_alloy", "Alloy"), ("af_aoede", "Aoede"), ("af_bella", "Bella"),
        ("af_heart", "Heart"), ("af_jessica", "Jessica"), ("af_kore", "Kore"),
        ("af_nicole", "Nicole"), ("af_nova", "Nova"), ("af_river", "River"),
        ("af_sarah", "Sarah"), ("af_sky", "Sky"), ("am_adam", "Adam"),
        ("am_echo", "Echo"), ("am_eric", "Eric"), ("am_fenrir", "Fenrir"),
        ("am_liam", "Liam"), ("am_michael", "Michael"), ("am_onyx", "Onyx"),
        ("am_puck", "Puck"), ("am_santa", "Santa Claus"), ("bf_alice", "Alice (britânica)"),
        ("bf_emma", "Emma (britânica)"), ("bf_isabella", "Isabella (britânica)"),
        ("bf_lily", "Lily (britânica)"), ("bm_daniel", "Daniel (britânica)"),
        ("bm_fable", "Fable (britânica)"), ("bm_george", "George (britânica)"),
        ("bm_lewis", "Lewis (britânica)"),
    ),
    "es": (("ef_dora", "Dora"), ("em_alex", "Alex"), ("em_santa", "Papai Noel")),
}
_VOICE_SPEEDS = {
    "slow": ("Lenta", 1.2),
    "normal": ("Normal", 1.0),
    "fast": ("Rápida", 0.85),
}
_VOICE_STYLES = {
    "steady": ("Estável", 0.45, 0.6),
    "natural": ("Natural", 0.667, 0.8),
    "expressive": ("Expressiva", 0.8, 1.0),
}
_VOICE_VOLUMES = {
    "soft": ("Suave", 0.65),
    "medium": ("Médio", 0.85),
    "high": ("Alto", 1.0),
}
_WHISPER_LANGUAGES = {"pt": "PT", "en": "EN", "es": "ES"}
_CUDA_DLL_DIRECTORY_HANDLES: list[Any] = []


def _load_llm_profiles() -> tuple[str, dict[str, dict[str, Any]]]:
    try:
        with _CONFIG_FILE.open("rb") as config_file:
            config = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Required configuration file not found: {_CONFIG_FILE}"
        ) from error

    llm_config = config.get("llm")
    if not isinstance(llm_config, dict):
        legacy = config.get("lm_studio", {})
        llm_config = {"provider": "lm_studio", **(legacy if isinstance(legacy, dict) else {})}

    providers = llm_config.get("providers")
    if isinstance(providers, dict) and providers:
        profiles = {
            str(provider).strip().lower(): values
            for provider, values in providers.items()
            if isinstance(values, dict)
        }
        if not profiles:
            raise ValueError(f"No valid provider profiles are configured in {_CONFIG_FILE}.")
        selected = str(llm_config.get("selected_provider", next(iter(profiles)))).strip().lower()
    else:
        # Accept the original single-provider layout for existing config files.
        selected = str(llm_config.get("provider", "lm_studio")).strip().lower()
        profiles = {
            selected: {
                key: value
                for key, value in llm_config.items()
                if key not in {"provider", "selected_provider"}
            }
        }

    if selected not in profiles:
        raise ValueError(
            f"Selected provider '{selected}' is not configured in [llm.providers]."
        )
    for provider, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"Provider profile '{provider}' must be a TOML table.")
        if not profile.get("base_url"):
            profile["base_url"] = _LLM_PROVIDER_URLS.get(provider, "")
        if not isinstance(profile.get("base_url"), str) or not profile["base_url"]:
            raise ValueError(f"Set base_url for provider '{provider}' in config.toml.")
        model = profile.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError(f"Set model as a string for provider '{provider}'.")
        if isinstance(model, str) and not model.strip():
            profile["model"] = None
        api_key = profile.get("api_key")
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError(f"Set api_key as a string for provider '{provider}'.")
        if api_key is None and isinstance(profile.get("api_key_env"), str):
            profile["api_key"] = os.getenv(profile["api_key_env"])
    return selected, profiles


def _load_tts_config() -> dict[str, Any]:
    with _CONFIG_FILE.open("rb") as config_file:
        config = tomllib.load(config_file)
    tts_config = config.get("tts", {})
    if not isinstance(tts_config, dict):
        raise ValueError("[tts] must be a TOML table in config.toml.")
    engine = tts_config.get("engine", "piper")
    if not isinstance(engine, str) or engine not in _TTS_ENGINES:
        raise ValueError("[tts].engine must be 'qwen3', 'kokoro' or 'piper'.")
    return {
        "engine": engine,
        "qwen3": _load_qwen3_tts_config(tts_config),
    }


def _load_qwen3_tts_config(tts_config: dict[str, Any]) -> dict[str, str]:
    qwen3 = tts_config.get("qwen3", {})
    if not isinstance(qwen3, dict):
        raise ValueError("[tts.qwen3] must be a TOML table.")
    model = qwen3.get("model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    device = qwen3.get("device", "cuda")
    reference_text = qwen3.get("reference_text", "")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("[tts.qwen3].model must be a model identifier.")
    if not isinstance(device, str) or device not in {"cuda", "cpu", "auto"}:
        raise ValueError("[tts.qwen3].device must be 'cuda', 'cpu' or 'auto'.")
    if not isinstance(reference_text, str):
        raise ValueError("[tts.qwen3].reference_text must be a string.")
    reference_audio = qwen3.get("reference_audio", "")
    if not isinstance(reference_audio, str):
        raise ValueError("[tts.qwen3].reference_audio must be a path string.")
    return {
        "model": model,
        "device": device,
        "reference_text": reference_text,
        "reference_audio": reference_audio,
    }


def _configure_cuda_runtime(device: str) -> None:
    if device.casefold() != "cuda" or os.name != "nt":
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    dll_directories = [
        site_packages / "nvidia" / "cublas" / "bin",
        site_packages / "nvidia" / "cuda_runtime" / "bin",
        site_packages / "nvidia" / "cudnn" / "bin",
        site_packages / "ctranslate2",
    ]
    for variable_name in ("CUDA_PATH", "CUDA_PATH_V12_9", "CUDA_PATH_V12_8"):
        cuda_path = os.getenv(variable_name)
        if cuda_path:
            dll_directories.append(Path(cuda_path) / "bin")

    toolkit_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if toolkit_root.is_dir():
        dll_directories.extend(toolkit_root.glob("v12.*/bin"))

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for directory in dll_directories:
        if not directory.is_dir():
            continue
        directory_text = str(directory)
        if directory_text.casefold() not in {
            entry.casefold() for entry in path_entries
        }:
            path_entries.insert(0, directory_text)
            _CUDA_DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory_text))
    os.environ["PATH"] = os.pathsep.join(path_entries)

    try:
        ctypes.WinDLL("cublas64_12.dll")
    except OSError as error:
        raise RuntimeError(
            "Whisper is configured to use the NVIDIA GPU, but CUDA 12 cuBLAS "
            "could not be loaded. Install the NVIDIA runtime dependencies from "
            "requirements-cuda.txt and restart the application."
        ) from error


def _ensure_kokoro_assets() -> tuple[str, str]:
    _KOKORO_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, url in _KOKORO_ASSETS.items():
        path = _KOKORO_MODEL_DIRECTORY / filename
        if not path.is_file():
            temporary_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
            _LOGGER.info("Downloading Kokoro asset %s.", filename)
            with httpx.stream(
                "GET", url, timeout=300.0, follow_redirects=True
            ) as response:
                response.raise_for_status()
                with temporary_path.open("wb") as asset_file:
                    for chunk in response.iter_bytes():
                        asset_file.write(chunk)
            temporary_path.replace(path)
        paths.append(str(path))
    return paths[0], paths[1]


def _qwen3_voice_prompt_cache_path(
    reference_audio: str, model: str, reference_text: str, device: str
) -> Path:
    fingerprint = hashlib.sha256()
    fingerprint.update(
        f"qwen3-voice-prompt-v1\0{model}\0{reference_text}\0{device}\0".encode()
    )
    with Path(reference_audio).open("rb") as reference_file:
        for chunk in iter(lambda: reference_file.read(1024 * 1024), b""):
            fingerprint.update(chunk)
    cache_dir = _USER_SETTINGS_FILE.parent / "qwen3" / "voice_prompts"
    return cache_dir / f"{fingerprint.hexdigest()}.pt"


@dataclass(frozen=True)
class LocalPipecatSettings:
    llm_provider: str = "lm_studio"
    llm_base_url: str = "http://localhost:1234/v1"
    llm_model: str | None = None
    llm_api_key: str | None = None
    whisper_model: str = "small"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    piper_voice: str = "pt_BR-faber-medium"
    audio_input_device: int | str | None = None
    audio_input_gain: float = 4.0
    audio_sample_rate: int = 16_000
    audio_channels: int = 1

    def __post_init__(self) -> None:
        if not self.llm_provider.strip():
            raise ValueError("llm_provider cannot be empty.")
        parsed_url = urlsplit(self.llm_base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
            raise ValueError("llm_base_url must be an absolute HTTP(S) URL.")
        if self.llm_provider in {"lm_studio", "ollama"} and parsed_url.hostname not in _LOCAL_HOSTS:
            raise ValueError(f"{self.llm_provider} must use a localhost URL.")
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
        if self.llm_model is not None and not self.llm_model.strip():
            raise ValueError("llm_model cannot be empty when provided.")
        if not self.whisper_device.strip() or not self.whisper_compute_type.strip():
            raise ValueError("Whisper device and compute type cannot be empty.")

    @classmethod
    def from_environment(cls) -> "LocalPipecatSettings":
        provider, profiles = _load_llm_profiles()
        profile = profiles[provider]

        input_device = os.getenv("AUDIO_INPUT_DEVICE")
        if input_device is not None:
            input_device = int(input_device) if input_device.isdecimal() else input_device

        return cls(
            llm_provider=provider,
            llm_base_url=profile["base_url"],
            llm_model=profile.get("model"),
            llm_api_key=profile.get("api_key"),
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
        if (
            not self._active
            or loop is None
            or loop.is_closed()
        ):
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
    def __init__(
        self,
        on_state_change: Callable[[AgentState], None],
        provider_name: str = "configured",
    ) -> None:
        super().__init__()
        self._on_state_change = on_state_change
        self._provider_name = provider_name

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, VADUserStartedSpeakingFrame):
                _LOGGER.info("VAD detected speech; waiting for the end of the utterance.")
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                _LOGGER.info("VAD detected the end of the utterance; Whisper will transcribe it.")
            elif isinstance(frame, LLMFullResponseStartFrame):
                _LOGGER.info(
                    "LLM provider '%s' started generating a response.",
                    self._provider_name,
                )
                self._on_state_change(AgentState.PROCESSING)
            elif isinstance(frame, TTSStartedFrame):
                self._on_state_change(AgentState.SPEAKING)
            elif isinstance(frame, TTSStoppedFrame):
                self._on_state_change(AgentState.LISTENING)
            elif isinstance(frame, LLMFullResponseEndFrame):
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
        elif isinstance(frame, ErrorFrame):
            _LOGGER.error("Pipecat pipeline reported an error: %s", frame.error)
        await self.push_frame(frame, direction)


class _Qwen3TTSService(TTSService):
    def __init__(
        self,
        *,
        model: Any,
        voice_clone_prompt: dict[str, Any],
        language: str,
        speed: float,
        volume: float,
    ) -> None:
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            stop_frame_timeout_s=30.0,
            settings=TTSSettings(model="qwen3-tts-12hz-0.6b-base", voice=None),
            sample_rate=24_000,
        )
        self._model = model
        self._voice_clone_prompt = voice_clone_prompt
        self._language = language
        self._speed = speed
        self._volume = volume

    @staticmethod
    def _synthesize_pcm(
        model: Any,
        voice_clone_prompt: dict[str, Any],
        text: str,
        language: str,
        speed: float,
        volume: float,
    ) -> tuple[bytes, int, float]:
        import numpy as np

        generation_started_at = time.perf_counter()
        waveforms, sample_rate = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
        )
        if not waveforms:
            raise RuntimeError("Qwen3-TTS returned no audio waveform.")
        waveform = np.asarray(waveforms[0], dtype=np.float32).squeeze()
        if waveform.ndim != 1 or waveform.size == 0:
            raise RuntimeError("Qwen3-TTS returned an empty or invalid waveform.")
        generation_seconds = time.perf_counter() - generation_started_at
        audio_seconds = waveform.size / sample_rate
        _LOGGER.info(
            "Qwen3-TTS generated %d characters in %.2f seconds "
            "(%.2fx realtime).",
            len(text),
            generation_seconds,
            audio_seconds / generation_seconds if generation_seconds else 0.0,
        )
        if speed != 1.0:
            import librosa

            waveform = librosa.effects.time_stretch(waveform, rate=speed)
        waveform = np.clip(waveform * volume, -1.0, 1.0)
        return (waveform * 32767.0).astype(np.int16).tobytes(), sample_rate, generation_seconds

    @traced_tts
    async def run_tts(
        self, text: str, context_id: str
    ) -> AsyncGenerator[Frame, None]:
        async def audio_chunks(audio: bytes):
            chunk_size = int(self.sample_rate * 0.08) * 2
            for offset in range(0, len(audio), chunk_size):
                yield audio[offset : offset + chunk_size]

        try:
            await self.start_tts_usage_metrics(text)
            audio, sample_rate, _ = await asyncio.to_thread(
                self._synthesize_pcm,
                self._model,
                self._voice_clone_prompt,
                text,
                self._language,
                self._speed,
                self._volume,
            )
            async for frame in self._stream_audio_frames_from_iterator(
                audio_chunks(audio), in_sample_rate=sample_rate, context_id=context_id
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as error:
            _LOGGER.exception("Qwen3-TTS speech synthesis failed.")
            yield ErrorFrame(error=f"Qwen3-TTS speech synthesis failed: {error}")
        finally:
            await self.stop_ttfb_metrics()


class _DiagnosticSileroVADAnalyzer(SileroVADAnalyzer):
    def __init__(self, *, sample_rate: int, params: VADParams) -> None:
        super().__init__(sample_rate=sample_rate, params=params)
        self._last_confidence = 0.0
        self._last_diagnostic_at = time.monotonic()

    def voice_confidence(self, buffer: bytes) -> float:
        confidence = super().voice_confidence(buffer)
        if hasattr(confidence, "item"):
            confidence = confidence.item()
        self._last_confidence = float(confidence)
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
        self._selected_llm_provider, self._llm_profiles = _load_llm_profiles()
        if self._settings.llm_provider not in self._llm_profiles:
            self._llm_profiles[self._settings.llm_provider] = {
                "base_url": self._settings.llm_base_url,
                "model": self._settings.llm_model,
                "api_key": self._settings.llm_api_key,
            }
        self._selected_llm_provider = self._settings.llm_provider
        tts_config = _load_tts_config()
        self._tts_engine = tts_config["engine"]
        qwen3_config = tts_config["qwen3"]
        reference_audio = qwen3_config["reference_audio"]
        self._qwen3_reference_audio = (
            str((_CONFIG_FILE.parent / reference_audio).resolve())
            if reference_audio and not Path(reference_audio).is_absolute()
            else reference_audio or None
        )
        self._qwen3_model_id = tts_config["qwen3"]["model"]
        self._qwen3_device = tts_config["qwen3"]["device"]
        self._qwen3_reference_text = qwen3_config["reference_text"]
        self._qwen3_model: Any | None = None
        self._qwen3_loaded_device: str | None = None
        self._audio_devices = self._discover_audio_input_devices()
        self._audio_input_device = self._resolve_initial_audio_input_device()
        conversation_settings = self._load_conversation_settings()
        configured_prompts = conversation_settings.get("prompts", {})
        self._system_prompts = (
            configured_prompts if isinstance(configured_prompts, dict) else {}
        )
        saved_language = conversation_settings.get("language")
        self._conversation_language = (
            saved_language if saved_language in _CONVERSATION_LANGUAGES else "pt"
        )
        saved_voices = conversation_settings.get("piper_voices", {})
        if not isinstance(saved_voices, dict):
            saved_voices = {}
        self._piper_voices_by_language = {
            language: list(voices) for language, voices in _PIPER_VOICES.items()
        }
        configured_voice = self._settings.piper_voice
        voice_prefixes = {
            "pt": "pt_BR-",
            "en": "en_US-",
            "es": ("es_ES-", "es_MX-"),
        }
        for language in self._piper_voices_by_language:
            configured_voice_matches_language = configured_voice.startswith(
                voice_prefixes[language]
            )
            configured_voice_is_listed = configured_voice in {
                voice for voice, _ in self._piper_voices_by_language[language]
            }
            if configured_voice_matches_language and not configured_voice_is_listed:
                self._piper_voices_by_language[language].append(
                    (configured_voice, configured_voice)
                )
        self._selected_voices_by_engine = {"piper": {}, "kokoro": {}}
        for language, voices in self._piper_voices_by_language.items():
            voice_ids = {voice for voice, _ in voices}
            selected = saved_voices.get(language)
            if selected not in voice_ids:
                selected = (
                    self._settings.piper_voice
                    if self._settings.piper_voice in voice_ids
                    else _CONVERSATION_LANGUAGES[language][1]
                )
            self._selected_voices_by_engine["piper"][language] = selected
        saved_kokoro_voices = conversation_settings.get("kokoro_voices", {})
        if not isinstance(saved_kokoro_voices, dict):
            saved_kokoro_voices = {}
        for language, voices in _KOKORO_VOICES.items():
            voice_ids = {voice for voice, _ in voices}
            selected = saved_kokoro_voices.get(language)
            self._selected_voices_by_engine["kokoro"][language] = (
                selected if selected in voice_ids else voices[0][0]
            )
        saved_voice = conversation_settings.get("voice", {})
        if not isinstance(saved_voice, dict):
            saved_voice = {}
        self._voice_speed = (
            saved_voice.get("speed")
            if saved_voice.get("speed") in _VOICE_SPEEDS
            else "normal"
        )
        self._voice_style = (
            saved_voice.get("style")
            if saved_voice.get("style") in _VOICE_STYLES
            else "natural"
        )
        self._voice_volume = (
            saved_voice.get("volume")
            if saved_voice.get("volume") in _VOICE_VOLUMES
            else "high"
        )
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
    def _load_saved_audio_settings() -> dict[str, Any]:
        try:
            settings = json.loads(_USER_SETTINGS_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        return settings if isinstance(settings, dict) else {}

    @staticmethod
    def _load_saved_audio_device() -> dict[str, str] | None:
        audio_device = LocalPipecatDataSource._load_saved_audio_settings().get(
            "audio_input_device"
        )
        if (
            isinstance(audio_device, dict)
            and isinstance(audio_device.get("name"), str)
            and isinstance(audio_device.get("host_api"), str)
        ):
            return audio_device
        return None

    @staticmethod
    def _save_user_settings(updates: dict[str, Any]) -> None:
        settings = LocalPipecatDataSource._load_saved_audio_settings()
        settings.update(updates)
        try:
            _USER_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            temporary_file = _USER_SETTINGS_FILE.with_suffix(".tmp")
            temporary_file.write_text(
                json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary_file.replace(_USER_SETTINGS_FILE)
        except OSError:
            _LOGGER.exception("Could not save user settings.")

    @staticmethod
    def _load_conversation_settings() -> dict[str, Any]:
        try:
            with _CONFIG_FILE.open("rb") as config_file:
                config = tomllib.load(config_file)
        except (OSError, tomllib.TOMLDecodeError):
            _LOGGER.exception(
                "Could not read conversation settings from %s.", _CONFIG_FILE
            )
            return {}
        conversation = config.get("conversation", {})
        return conversation if isinstance(conversation, dict) else {}

    @staticmethod
    def _save_conversation_settings(updates: dict[str, Any]) -> None:
        try:
            contents = _CONFIG_FILE.read_text(encoding="utf-8")
            lines = contents.splitlines()
            section_start = next(
                (
                    index
                    for index, line in enumerate(lines)
                    if line.strip() == "[conversation]"
                ),
                None,
            )
            if section_start is None:
                if lines and lines[-1].strip():
                    lines.append("")
                lines.append("[conversation]")
                section_start = len(lines) - 1
                section_end = len(lines)
            else:
                section_end = next(
                    (
                        index
                        for index in range(section_start + 1, len(lines))
                        if lines[index].lstrip().startswith("[")
                    ),
                    len(lines),
                )

            existing = LocalPipecatDataSource._load_conversation_settings()
            existing.update(updates)
            selected_voices = existing.get("piper_voices", {})
            if not isinstance(selected_voices, dict):
                selected_voices = {}
            kokoro_voices = existing.get("kokoro_voices", {})
            if not isinstance(kokoro_voices, dict):
                kokoro_voices = {}
            voice_settings = existing.get("voice", {})
            if not isinstance(voice_settings, dict):
                voice_settings = {}
            serialized = {
                "language": json.dumps(existing["language"], ensure_ascii=False),
                "piper_voices": "{ "
                + ", ".join(
                    f"{language} = {json.dumps(voice, ensure_ascii=False)}"
                    for language, voice in selected_voices.items()
                )
                + " }",
                "kokoro_voices": "{ "
                + ", ".join(
                    f"{language} = {json.dumps(voice, ensure_ascii=False)}"
                    for language, voice in kokoro_voices.items()
                )
                + " }",
                "voice": "{ "
                + ", ".join(
                    f"{key} = {json.dumps(value, ensure_ascii=False)}"
                    for key, value in voice_settings.items()
                )
                + " }",
            }
            for key, value in serialized.items():
                setting_index = next(
                    (
                        index
                        for index in range(section_start + 1, section_end)
                        if lines[index].partition("=")[0].strip() == key
                    ),
                    None,
                )
                setting_line = f"{key} = {value}"
                if setting_index is None:
                    lines.insert(section_end, setting_line)
                    section_end += 1
                else:
                    lines[setting_index] = setting_line

            temporary_file = _CONFIG_FILE.with_suffix(".toml.tmp")
            temporary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            temporary_file.replace(_CONFIG_FILE)
        except (OSError, KeyError, TypeError):
            _LOGGER.exception(
                "Could not save conversation settings to %s.", _CONFIG_FILE
            )

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
        self._save_user_settings(
            {
                "audio_input_device": {
                    "name": selected_device["name"],
                    "host_api": selected_device["host_api"],
                }
            }
        )

    def get_conversation_languages(self) -> list[tuple[str, str]]:
        return [(code, details[0]) for code, details in _CONVERSATION_LANGUAGES.items()]

    def get_tts_engines(self) -> list[tuple[str, str]]:
        return list(_TTS_ENGINES.items())

    def get_selected_tts_engine(self) -> str:
        return self._tts_engine

    def get_qwen3_reference_audio(self) -> str | None:
        return self._qwen3_reference_audio

    def set_qwen3_reference_audio(self, reference_path: str) -> None:
        path = Path(reference_path).expanduser().resolve()
        if not path.is_file() or path.suffix.casefold() != ".wav":
            raise ValueError("Escolha um arquivo WAV existente para a voz de referência.")
        try:
            config_path = str(path.relative_to(_CONFIG_FILE.parent))
        except ValueError:
            config_path = str(path)
        self._save_toml_setting("tts.qwen3", "reference_audio", config_path)
        self._qwen3_reference_audio = str(path)

    def select_tts_engine(self, engine: str) -> None:
        if engine not in _TTS_ENGINES:
            raise ValueError(f"Unsupported TTS engine: {engine}")
        self._save_toml_setting("tts", "engine", engine)
        self._tts_engine = engine

    @staticmethod
    def _save_toml_setting(section: str, key: str, value: str) -> None:
        lines = _CONFIG_FILE.read_text(encoding="utf-8").splitlines()
        section_header = f"[{section}]"
        section_start = next(
            (index for index, line in enumerate(lines) if line.strip() == section_header),
            None,
        )
        if section_start is None:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append(section_header)
            section_start = len(lines) - 1
            section_end = len(lines)
        else:
            section_end = next(
                (
                    index
                    for index in range(section_start + 1, len(lines))
                    if lines[index].lstrip().startswith("[")
                ),
                len(lines),
            )
        setting_line = f"{key} = {json.dumps(value, ensure_ascii=False)}"
        setting_index = next(
            (
                index
                for index in range(section_start + 1, section_end)
                if lines[index].partition("=")[0].strip() == key
            ),
            None,
        )
        if setting_index is None:
            lines.insert(section_end, setting_line)
        else:
            lines[setting_index] = setting_line
        temporary_file = _CONFIG_FILE.with_suffix(".toml.tmp")
        temporary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary_file.replace(_CONFIG_FILE)

    def get_llm_providers(self) -> list[tuple[str, str]]:
        return [
            (
                provider,
                _LLM_PROVIDER_LABELS.get(provider, provider.replace("_", " ").title()),
            )
            for provider in self._llm_profiles
        ]

    def get_selected_llm_provider(self) -> str:
        return self._selected_llm_provider

    def select_llm_provider(self, provider: str) -> None:
        provider = provider.strip().lower()
        profile = self._llm_profiles.get(provider)
        if profile is None:
            raise ValueError(f"LLM provider {provider!r} is not configured.")
        updated_settings = replace(
            self._settings,
            llm_provider=provider,
            llm_base_url=profile["base_url"],
            llm_model=profile.get("model"),
            llm_api_key=profile.get("api_key"),
        )
        self._save_selected_llm_provider(provider)
        self._settings = updated_settings
        self._selected_llm_provider = provider

    @staticmethod
    def _save_selected_llm_provider(provider: str) -> None:
        lines = _CONFIG_FILE.read_text(encoding="utf-8").splitlines()
        section_start = next(
            (index for index, line in enumerate(lines) if line.strip() == "[llm]"),
            None,
        )
        if section_start is None:
            if lines and lines[-1].strip():
                lines.append("")
            lines.append("[llm]")
            section_start = len(lines) - 1
            section_end = len(lines)
        else:
            section_end = next(
                (
                    index
                    for index in range(section_start + 1, len(lines))
                    if lines[index].lstrip().startswith("[")
                ),
                len(lines),
            )

        selected_line = f"selected_provider = {json.dumps(provider)}"
        selected_index = next(
            (
                index
                for index in range(section_start + 1, section_end)
                if lines[index].partition("=")[0].strip() == "selected_provider"
            ),
            None,
        )
        if selected_index is None:
            lines.insert(section_end, selected_line)
        else:
            lines[selected_index] = selected_line

        temporary_file = _CONFIG_FILE.with_suffix(".toml.tmp")
        temporary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary_file.replace(_CONFIG_FILE)

    def get_selected_conversation_language(self) -> str:
        return self._conversation_language

    def get_voices_by_language(
        self, engine: str | None = None
    ) -> dict[str, list[tuple[str, str]]]:
        selected_engine = engine or self._tts_engine
        voices_by_language = (
            _KOKORO_VOICES
            if selected_engine == "kokoro"
            else self._piper_voices_by_language
        )
        return {
            language: list(voices)
            for language, voices in voices_by_language.items()
        }

    def get_selected_voice(self) -> str:
        engine = self._voice_engine_key()
        return self._selected_voices_by_engine[engine][self._conversation_language]

    def get_selected_voices_by_language(self) -> dict[str, str]:
        return dict(self._selected_voices_by_engine[self._voice_engine_key()])

    def _voice_engine_key(self) -> str:
        return "kokoro" if self._tts_engine == "kokoro" else "piper"

    def get_voice_speed_options(self) -> list[tuple[str, str]]:
        return [(key, value[0]) for key, value in _VOICE_SPEEDS.items()]

    def get_voice_style_options(self) -> list[tuple[str, str]]:
        return [(key, value[0]) for key, value in _VOICE_STYLES.items()]

    def get_voice_volume_options(self) -> list[tuple[str, str]]:
        return [(key, value[0]) for key, value in _VOICE_VOLUMES.items()]

    def get_voice_adjustments(self) -> dict[str, str]:
        return {
            "speed": self._voice_speed,
            "style": self._voice_style,
            "volume": self._voice_volume,
        }

    def _save_voice_adjustments(self) -> None:
        conversation = self._load_conversation_settings()
        self._save_conversation_settings(
            {
                "language": self._conversation_language,
                "piper_voices": self._selected_voices_by_engine["piper"],
                "kokoro_voices": self._selected_voices_by_engine["kokoro"],
                "voice": {
                    **(
                        conversation.get("voice", {})
                        if isinstance(conversation.get("voice"), dict)
                        else {}
                    ),
                    "speed": self._voice_speed,
                    "style": self._voice_style,
                    "volume": self._voice_volume,
                },
            }
        )

    def select_voice_speed(self, speed: str) -> None:
        if speed not in _VOICE_SPEEDS:
            raise ValueError(f"Unsupported voice speed: {speed}")
        self._voice_speed = speed
        self._save_voice_adjustments()

    def select_voice_style(self, style: str) -> None:
        if style not in _VOICE_STYLES:
            raise ValueError(f"Unsupported voice style: {style}")
        self._voice_style = style
        self._save_voice_adjustments()

    def select_voice_volume(self, volume: str) -> None:
        if volume not in _VOICE_VOLUMES:
            raise ValueError(f"Unsupported voice volume: {volume}")
        self._voice_volume = volume
        self._save_voice_adjustments()

    def select_conversation_language(self, language: str) -> None:
        if language not in _CONVERSATION_LANGUAGES:
            raise ValueError(f"Unsupported conversation language: {language}")
        self._conversation_language = language
        self._save_conversation_settings(
            {
                "language": language,
                "piper_voices": self._selected_voices_by_engine["piper"],
                "kokoro_voices": self._selected_voices_by_engine["kokoro"],
            }
        )

    def select_voice(self, voice: str) -> None:
        engine = self._voice_engine_key()
        voices_by_language = (
            _KOKORO_VOICES
            if engine == "kokoro"
            else self._piper_voices_by_language
        )
        available_voices = {
            voice_id
            for voice_id, _ in voices_by_language[self._conversation_language]
        }
        if voice not in available_voices:
            raise ValueError(
                f"Voice {voice!r} is not available for language "
                f"{self._conversation_language!r}."
            )
        self._selected_voices_by_engine[engine][self._conversation_language] = voice
        self._save_conversation_settings(
            {
                "language": self._conversation_language,
                "piper_voices": self._selected_voices_by_engine["piper"],
                "kokoro_voices": self._selected_voices_by_engine["kokoro"],
            }
        )

    def set_state_listener(
        self, listener: Callable[[AgentState], None] | None
    ) -> None:
        self._on_state_change = listener

    async def start_session(self) -> None:
        async with self._session_lock:
            if self._runner_task is not None and not self._runner_task.done():
                raise RuntimeError("A Pipecat voice session is already running.")
            await self._clear_finished_session()
            session_llm_settings = self._settings

            system_prompt = self._system_prompts.get(self._conversation_language)
            if not isinstance(system_prompt, str) or not system_prompt.strip():
                raise ValueError(
                    "Set a non-empty prompt for the selected language in "
                    "[conversation.prompts] in config.toml."
                )

            _configure_cuda_runtime(self._settings.whisper_device)

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
            from pipecat.utils.tracing.service_decorators import traced_tts
            from pipecat.workers.runner import WorkerRunner
            from piper.config import SynthesisConfig

            class _ConfigurablePiperTTSService(PiperTTSService):
                def __init__(self, *, synthesis_config: SynthesisConfig, **kwargs):
                    self._synthesis_config = synthesis_config
                    super().__init__(**kwargs)

                @traced_tts
                async def run_tts(
                    self, text: str, context_id: str
                ) -> AsyncGenerator[Frame, None]:
                    def next_chunk(iterator: Iterator[Any]) -> Any | None:
                        try:
                            return next(iterator)
                        except StopIteration:
                            return None

                    async def audio_chunks(iterator: Iterable[Any]):
                        while True:
                            chunk = await asyncio.to_thread(next_chunk, iterator)
                            if chunk is None:
                                return
                            yield chunk.audio_int16_bytes

                    try:
                        await self.start_tts_usage_metrics(text)
                        async for frame in self._stream_audio_frames_from_iterator(
                            audio_chunks(
                                self._voice.synthesize(
                                    text, self._synthesis_config
                                )
                            ),
                            in_sample_rate=self._voice.config.sample_rate,
                            context_id=context_id,
                        ):
                            await self.stop_ttfb_metrics()
                            yield frame
                    except Exception as error:
                        _LOGGER.exception("Piper speech synthesis failed.")
                        yield ErrorFrame(
                            error=f"Piper speech synthesis failed: {error}"
                        )
                    finally:
                        await self.stop_ttfb_metrics()

            model = await self._resolve_llm_model(session_llm_settings)
            api_key = self._resolve_llm_api_key(session_llm_settings)
            llm = OpenAILLMService(
                settings=OpenAILLMService.Settings(model=model),
                api_key=api_key,
                base_url=session_llm_settings.llm_base_url,
            )
            stt = WhisperSTTService(
                device=self._settings.whisper_device,
                compute_type=self._settings.whisper_compute_type,
                settings=WhisperSTTService.Settings(
                    model=self._settings.whisper_model,
                    language=getattr(Language, _WHISPER_LANGUAGES[self._conversation_language]),
                    no_speech_prob=1.0,
                ),
                sample_rate=self._settings.audio_sample_rate,
                num_channels=self._settings.audio_channels,
            )
            if self._tts_engine == "piper":
                tts = _ConfigurablePiperTTSService(
                    settings=PiperTTSService.Settings(voice=self.get_selected_voice()),
                    synthesis_config=SynthesisConfig(
                        length_scale=_VOICE_SPEEDS[self._voice_speed][1],
                        noise_scale=_VOICE_STYLES[self._voice_style][1],
                        noise_w_scale=_VOICE_STYLES[self._voice_style][2],
                        volume=_VOICE_VOLUMES[self._voice_volume][1],
                    ),
                )
            elif self._tts_engine == "kokoro":
                try:
                    from pipecat.services.kokoro.tts import KokoroTTSService
                except ImportError as error:
                    raise RuntimeError(
                        "Kokoro dependencies are missing. Install the updated "
                        "project requirements with `uv pip install --python "
                        ".venv\\Scripts\\python.exe -r requirements.txt`."
                    ) from error
                kokoro_languages = {
                    "pt": Language.PT_BR,
                    "en": Language.EN_US,
                    "es": Language.ES,
                }
                _LOGGER.info(
                    "Loading Kokoro-82M model and voice assets on first use."
                )
                model_path, voices_path = await asyncio.to_thread(
                    _ensure_kokoro_assets
                )
                tts = await asyncio.to_thread(
                    KokoroTTSService,
                    model_path=model_path,
                    voices_path=voices_path,
                    settings=KokoroTTSService.Settings(
                        model="kokoro-82m",
                        voice=self.get_selected_voice(),
                        language=kokoro_languages[self._conversation_language],
                        speed=_QWEN3_SPEED[self._voice_speed],
                    ),
                    sample_rate=24_000,
                )
            elif self._tts_engine == "qwen3":
                if not self._qwen3_reference_audio or not Path(
                    self._qwen3_reference_audio
                ).is_file():
                    raise FileNotFoundError(
                        "Qwen3-TTS requires a configured WAV voice reference. "
                        "Add one from the tray menu under Voice reference."
                    )
                import torch
                from qwen_tts import Qwen3TTSModel

                device = self._qwen3_device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                elif device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError(
                        "Qwen3-TTS is configured for CUDA, but PyTorch cannot "
                        "access the NVIDIA GPU. Install the CUDA-enabled Qwen3 "
                        "environment or set [tts.qwen3].device = 'cpu'."
                    )
                if self._qwen3_model is None or self._qwen3_loaded_device != device:
                    if device == "cuda":
                        dtype = (
                            torch.bfloat16
                            if torch.cuda.is_bf16_supported()
                            else torch.float16
                        )
                    else:
                        dtype = torch.float32
                    _LOGGER.info(
                        "Loading Qwen3-TTS model %s on %s with %s.",
                        self._qwen3_model_id,
                        device,
                        dtype,
                    )
                    self._qwen3_model = await asyncio.to_thread(
                        Qwen3TTSModel.from_pretrained,
                        self._qwen3_model_id,
                        device_map="cuda:0" if device == "cuda" else "cpu",
                        dtype=dtype,
                        attn_implementation="sdpa",
                    )
                    self._qwen3_loaded_device = device

                prompt_cache_path = await asyncio.to_thread(
                    _qwen3_voice_prompt_cache_path,
                    self._qwen3_reference_audio,
                    self._qwen3_model_id,
                    self._qwen3_reference_text,
                    device,
                )
                voice_clone_prompt = None
                if prompt_cache_path.is_file():
                    try:
                        voice_clone_prompt = await asyncio.to_thread(
                            torch.load,
                            prompt_cache_path,
                            map_location=device,
                            weights_only=True,
                        )
                        _LOGGER.info(
                            "Reused cached Qwen3-TTS voice reference prompt."
                        )
                    except Exception as error:
                        _LOGGER.warning(
                            "Could not load cached Qwen3-TTS voice prompt; "
                            "reprocessing the WAV: %s",
                            error,
                        )
                if voice_clone_prompt is None:
                    x_vector_only = not bool(self._qwen3_reference_text.strip())
                    prompt_items = await asyncio.to_thread(
                        self._qwen3_model.create_voice_clone_prompt,
                        ref_audio=self._qwen3_reference_audio,
                        ref_text=self._qwen3_reference_text or None,
                        x_vector_only_mode=x_vector_only,
                    )
                    if not prompt_items:
                        raise RuntimeError(
                            "Qwen3-TTS could not prepare the selected voice reference."
                        )
                    prompt_item = prompt_items[0]
                    voice_clone_prompt = {
                        "ref_code": [prompt_item.ref_code],
                        "ref_spk_embedding": [prompt_item.ref_spk_embedding],
                        "x_vector_only_mode": [prompt_item.x_vector_only_mode],
                        "icl_mode": [prompt_item.icl_mode],
                    }
                    try:
                        prompt_cache_path.parent.mkdir(parents=True, exist_ok=True)
                        temporary_cache_path = prompt_cache_path.with_name(
                            f"{prompt_cache_path.stem}.{os.getpid()}.tmp"
                        )
                        await asyncio.to_thread(
                            torch.save, voice_clone_prompt, temporary_cache_path
                        )
                        temporary_cache_path.replace(prompt_cache_path)
                        if x_vector_only:
                            _LOGGER.warning(
                                "Qwen3-TTS is using speaker-only cloning because "
                                "[tts.qwen3].reference_text is empty. Add the WAV "
                                "transcript there for higher voice similarity."
                            )
                        else:
                            _LOGGER.info("Cached processed Qwen3-TTS voice prompt.")
                    except Exception as error:
                        _LOGGER.warning(
                            "Could not cache Qwen3-TTS voice prompt: %s", error
                        )
                qwen_languages = {
                    "pt": "Portuguese",
                    "en": "English",
                    "es": "Spanish",
                }
                tts = _Qwen3TTSService(
                    model=self._qwen3_model,
                    voice_clone_prompt=voice_clone_prompt,
                    language=qwen_languages[self._conversation_language],
                    speed=_QWEN3_SPEED[self._voice_speed],
                    volume=_VOICE_VOLUMES[self._voice_volume][1],
                )


            context = LLMContext(
                [
                    {
                        "role": "system",
                        "content": system_prompt.strip(),
                    }
                ]
            )
            context_aggregator = LLMContextAggregatorPair(
                context,
                user_params=LLMUserAggregatorParams(
                    user_turn_strategies=UserTurnStrategies(
                        start=[
                            VADUserTurnStartStrategy(enable_interruptions=True),
                            TranscriptionUserTurnStartStrategy(
                                enable_interruptions=True
                            ),
                        ]
                    ),
                    empty_user_turn=None,
                ),
            )
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
            state_processor = _AgentStateProcessor(
                self._notify_state, session_llm_settings.llm_provider
            )
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
                    _LOGGER.info(
                        "Waiting for the current transcription and response to finish before stopping the session."
                    )
                    drained = await pipeline_worker.flush_pipeline(
                        timeout=_SESSION_DRAIN_TIMEOUT_SECS
                    )
                    if not drained:
                        _LOGGER.warning(
                            "The voice turn did not finish within %.0f seconds; stopping the pipeline anyway.",
                            _SESSION_DRAIN_TIMEOUT_SECS,
                        )
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

    async def _resolve_llm_model(
        self, settings: LocalPipecatSettings | None = None
    ) -> str:
        settings = settings or self._settings
        configured_model = settings.llm_model
        if configured_model:
            return configured_model

        if settings.llm_provider != "lm_studio":
            raise RuntimeError(
                "A model must be configured for this provider in [llm]."
            )

        endpoint = f"{settings.llm_base_url.rstrip('/')}/models"
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(endpoint)
            response.raise_for_status()

        payload = response.json()
        models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise RuntimeError(
                "The local provider returned an invalid /v1/models response; "
                "set model in [llm] to select a loaded model."
            )
        for model in models:
            if isinstance(model, dict) and isinstance(model.get("id"), str):
                model_id = model["id"].strip()
                if model_id:
                    return model_id

        raise RuntimeError(
            "No model is loaded in LM Studio. Load a model or set model in [llm]."
        )

    def _resolve_llm_api_key(
        self, settings: LocalPipecatSettings | None = None
    ) -> str:
        settings = settings or self._settings
        if settings.llm_provider in {"lm_studio", "ollama"}:
            return settings.llm_provider
        api_key = settings.llm_api_key
        if not api_key:
            raise ValueError(
                f"Configure api_key for provider '{settings.llm_provider}' "
                "in config.toml."
            )
        return api_key

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
