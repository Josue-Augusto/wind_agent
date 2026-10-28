from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import math
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

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
_KOKORO_VOICE_DEFAULTS = {
    "speed": "1.0",
    "sentence_pause": "0.25",
    "clause_pause": "0.1",
    "trim": "true",
}
_KOKORO_VOICE_OPTIONS = {
    "speed": tuple(
        (f"{step / 10:.1f}", f"{step / 10:.1f}x".replace(".", ","))
        for step in range(5, 21)
    ),
    "sentence_pause": tuple(
        (value, f"{value.replace('.', ',')} s")
        for value in ("0", "0.1", "0.25", "0.5", "0.75", "1.0")
    ),
    "clause_pause": tuple(
        (value, f"{value.replace('.', ',')} s")
        for value in ("0", "0.05", "0.1", "0.2", "0.3", "0.5")
    ),
    "trim": (("true", "Ativado"), ("false", "Desativado")),
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


