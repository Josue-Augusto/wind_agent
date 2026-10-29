from __future__ import annotations

import asyncio
import json
import logging
import os
import tomllib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, TYPE_CHECKING

import httpx
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from src.data.datasources.local_pipecat_config import (
    _CONFIG_FILE,
    _CONVERSATION_LANGUAGES,
    _KOKORO_VOICES,
    _KOKORO_VOICE_DEFAULTS,
    _KOKORO_VOICE_OPTIONS,
    _LLM_PROVIDER_LABELS,
    _PIPER_VOICES,
    _SESSION_DRAIN_TIMEOUT_SECS,
    _TTS_ENGINES,
    _VOICE_SPEEDS,
    _VOICE_STYLES,
    _VOICE_VOLUMES,
    _WHISPER_LANGUAGES,
    LocalPipecatSettings,
    _configure_cuda_runtime,
    _load_llm_profiles,
    _load_tts_config,
)
from src.data.datasources.local_audio_input_datasource import LocalAudioInputDataSource
from src.data.datasources.local_tts_service_factory import LocalTTSServiceFactory
from src.data.datasources.pipecat_processors import (
    _AgentStateProcessor,
    _DiagnosticSileroVADAnalyzer,
    _SoundDeviceInputProcessor,
    _SoundDeviceOutputProcessor,
    _TranscriptionLoggingProcessor,
)
from src.domain.entities.agent_state import AgentState

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.workers.runner import WorkerRunner

_LOGGER = logging.getLogger(__name__)


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
        self._tts_service_factory = LocalTTSServiceFactory()
        self._push_to_talk_enabled = False
        self._push_to_talk_pressed = False
        self._audio_input = LocalAudioInputDataSource(self._settings)
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
        saved_kokoro_settings = conversation_settings.get("kokoro_voice", {})
        if not isinstance(saved_kokoro_settings, dict):
            saved_kokoro_settings = {}
        self._kokoro_voice_settings = dict(_KOKORO_VOICE_DEFAULTS)
        for key, value in saved_kokoro_settings.items():
            if key not in _KOKORO_VOICE_OPTIONS:
                continue
            value = str(value).lower() if key == "trim" else str(value)
            if value in {option for option, _ in _KOKORO_VOICE_OPTIONS[key]}:
                self._kokoro_voice_settings[key] = value
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
            kokoro_voice_settings = existing.get("kokoro_voice", {})
            if not isinstance(kokoro_voice_settings, dict):
                kokoro_voice_settings = {}
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
                "kokoro_voice": "{ "
                + ", ".join(
                    f"{key} = {str(value).lower() if key == 'trim' else value}"
                    for key, value in kokoro_voice_settings.items()
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
        return self._audio_input.get_selected_audio_input_device()

    def get_audio_input_devices(self) -> list[tuple[int, str]]:
        return self._audio_input.get_audio_input_devices()

    def set_push_to_talk(self, enabled: bool) -> None:
        self._push_to_talk_enabled = enabled
        if not enabled:
            self._push_to_talk_pressed = False
        if self._input_processor is not None:
            self._input_processor.set_push_to_talk(
                enabled=enabled, pressed=self._push_to_talk_pressed
            )

    def is_push_to_talk_enabled(self) -> bool:
        return self._push_to_talk_enabled

    def set_push_to_talk_pressed(self, pressed: bool) -> None:
        self._push_to_talk_pressed = pressed
        if self._input_processor is not None:
            self._input_processor.set_push_to_talk_pressed(pressed)

    def get_audio_input_capture_channels(self) -> int:
        return self._audio_input.get_capture_channels()

    def select_audio_input_device(self, device_index: int) -> None:
        self._audio_input.select_audio_input_device(device_index)

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

    def get_kokoro_voice_options(self) -> dict[str, list[tuple[str, str]]]:
        return {
            key: list(options) for key, options in _KOKORO_VOICE_OPTIONS.items()
        }

    def get_kokoro_voice_settings(self) -> dict[str, str]:
        return dict(self._kokoro_voice_settings)

    def select_kokoro_voice_setting(self, setting: str, value: str) -> None:
        valid_values = {
            option for option, _ in _KOKORO_VOICE_OPTIONS.get(setting, ())
        }
        if value not in valid_values:
            raise ValueError(
                f"Unsupported Kokoro voice setting: {setting}={value!r}."
            )
        self._kokoro_voice_settings[setting] = value
        self._save_conversation_settings(
            {
                "language": self._conversation_language,
                "piper_voices": self._selected_voices_by_engine["piper"],
                "kokoro_voices": self._selected_voices_by_engine["kokoro"],
                "kokoro_voice": self._kokoro_voice_settings,
            }
        )

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
            from pipecat.services.whisper.stt import WhisperSTTService
            from pipecat.transcriptions.language import Language
            from pipecat.workers.runner import WorkerRunner

            model = await self._resolve_llm_model(session_llm_settings)
            api_key = self._resolve_llm_api_key(session_llm_settings)
            if session_llm_settings.llm_provider == "lm_studio":
                profile = self._llm_profiles.get("lm_studio", {})
                mcp_servers = profile.get("mcp_servers", [])
                if not isinstance(mcp_servers, list) or any(
                    not isinstance(server, str) or not server.strip()
                    for server in mcp_servers
                ):
                    raise ValueError(
                        "[llm.providers.lm_studio].mcp_servers must be an array "
                        "of LM Studio MCP server names."
                    )
                if mcp_servers:
                    from src.data.datasources.local_lm_studio_mcp_service import (
                        LocalLMStudioMCPService,
                    )
                    llm = LocalLMStudioMCPService(
                        model=model,
                        base_url=session_llm_settings.llm_base_url,
                        api_key=api_key,
                        mcp_servers=mcp_servers,
                        language=self._conversation_language,
                    )
                else:
                    llm = OpenAILLMService(
                        settings=OpenAILLMService.Settings(model=model),
                        api_key=api_key,
                        base_url=session_llm_settings.llm_base_url,
                    )
            else:
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
            tts = await self._tts_service_factory.create(
                engine=self._tts_engine,
                voice=self.get_selected_voice(),
                language=self._conversation_language,
                voice_speed=self._voice_speed,
                voice_style=self._voice_style,
                voice_volume=self._voice_volume,
                kokoro_voice_settings=self._kokoro_voice_settings,
                qwen3_reference_audio=self._qwen3_reference_audio,
                qwen3_model_id=self._qwen3_model_id,
                qwen3_device=self._qwen3_device,
                qwen3_reference_text=self._qwen3_reference_text,
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
                                enable_interruptions=False
                            ),
                        ]
                    ),
                    empty_user_turn=None,
                ),
            )
            input_processor = _SoundDeviceInputProcessor(
                self._settings.audio_sample_rate,
                self._settings.audio_channels,
                self._audio_input.get_selected_audio_input_device(),
                self.get_audio_input_capture_channels(),
                self._settings.audio_input_gain,
            )
            vad_processor = VADProcessor(
                vad_analyzer=_DiagnosticSileroVADAnalyzer(
                    sample_rate=self._settings.audio_sample_rate,
                    params=VADParams(
                        confidence=0.75,
                        start_secs=0.25,
                        min_volume=0.25,
                        stop_secs=0.8,
                    ),
                )
            )
            output_processor = _SoundDeviceOutputProcessor()
            input_processor.set_push_to_talk(
                enabled=self._push_to_talk_enabled,
                pressed=self._push_to_talk_pressed,
            )
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
            _LOGGER.info("Voice session is listening continuously.")
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
        api_key = settings.llm_api_key
        if api_key:
            return api_key
        if settings.llm_provider in {"lm_studio", "ollama"}:
            return settings.llm_provider
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
