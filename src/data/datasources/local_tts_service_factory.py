from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Iterable, Iterator
from pathlib import Path
from typing import Any

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.utils.tracing.service_decorators import traced_tts
from src.data.datasources.local_pipecat_config import (
    _QWEN3_SPEED,
    _VOICE_SPEEDS,
    _VOICE_STYLES,
    _VOICE_VOLUMES,
    _ensure_kokoro_assets,
    _qwen3_voice_prompt_cache_path,
)
from src.data.datasources.pipecat_processors import _Qwen3TTSService

_LOGGER = logging.getLogger(__name__)

class LocalTTSServiceFactory:
    """Create the selected local TTS backend and retain reusable Qwen model state."""

    def __init__(self) -> None:
        self._qwen3_model: Any | None = None
        self._qwen3_loaded_device: str | None = None

    async def create(
        self,
        *,
        engine: str,
        voice: str,
        language: str,
        voice_speed: str,
        voice_style: str,
        voice_volume: str,
        kokoro_voice_settings: dict[str, str],
        qwen3_reference_audio: str | None,
        qwen3_model_id: str,
        qwen3_device: str,
        qwen3_reference_text: str,
    ) -> Any:
        from pipecat.services.piper.tts import PiperTTSService
        from pipecat.transcriptions.language import Language
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
        if engine == "piper":
            tts = _ConfigurablePiperTTSService(
                settings=PiperTTSService.Settings(voice=voice),
                synthesis_config=SynthesisConfig(
                    length_scale=_VOICE_SPEEDS[voice_speed][1],
                    noise_scale=_VOICE_STYLES[voice_style][1],
                    noise_w_scale=_VOICE_STYLES[voice_style][2],
                    volume=_VOICE_VOLUMES[voice_volume][1],
                ),
            )
        elif engine == "kokoro":
            try:
                from pipecat.services.kokoro.tts import KokoroTTSService
            except ImportError as error:
                raise RuntimeError(
                    "Kokoro dependencies are missing. Install the updated "
                    "project requirements with `uv pip install --python "
                    ".venv\\Scripts\\python.exe -r requirements.txt`."
                ) from error

            class _ConfiguredKokoroTTSService(KokoroTTSService):
                def __init__(
                    self,
                    *,
                    trim: bool,
                    sentence_pause: float,
                    clause_pause: float,
                    **kwargs: Any,
                ) -> None:
                    self._trim = trim
                    self._sentence_pause = sentence_pause
                    self._clause_pause = clause_pause
                    super().__init__(**kwargs)

                @traced_tts
                async def run_tts(
                    self, text: str, context_id: str
                ) -> AsyncGenerator[Frame, None]:
                    import numpy as np

                    try:
                        await self.start_tts_usage_metrics(text)
                        stream = self._kokoro.create_stream(
                            text,
                            voice=self._settings.voice,
                            lang=self.language_to_service_language(
                                self._settings.language
                            ),
                            speed=self._settings.speed,
                            trim=self._trim,
                            sentence_pause=self._sentence_pause,
                            clause_pause=self._clause_pause,
                        )
                        async for samples, sample_rate in stream:
                            await self.stop_ttfb_metrics()
                            audio = (samples * 32767).astype(np.int16).tobytes()
                            audio = await self._resampler.resample(
                                audio, sample_rate, self.sample_rate
                            )
                            yield TTSAudioRawFrame(
                                audio=audio,
                                sample_rate=self.sample_rate,
                                num_channels=1,
                                context_id=context_id,
                            )
                    except Exception as error:
                        _LOGGER.exception("Kokoro-82M speech synthesis failed.")
                        yield ErrorFrame(
                            error=f"Kokoro-82M speech synthesis failed: {error}"
                        )
                    finally:
                        await self.stop_ttfb_metrics()

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
                _ConfiguredKokoroTTSService,
                model_path=model_path,
                voices_path=voices_path,
                trim=kokoro_voice_settings["trim"] == "true",
                sentence_pause=float(
                    kokoro_voice_settings["sentence_pause"]
                ),
                clause_pause=float(kokoro_voice_settings["clause_pause"]),
                settings=KokoroTTSService.Settings(
                    model="kokoro-82m",
                    voice=voice,
                    language=kokoro_languages[language],
                    speed=_QWEN3_SPEED[voice_speed],
                ),
                sample_rate=24_000,
            )
        elif engine == "qwen3":
            if not qwen3_reference_audio or not Path(
                qwen3_reference_audio
            ).is_file():
                raise FileNotFoundError(
                    "Qwen3-TTS requires a configured WAV voice reference. "
                    "Add one from the tray menu under Voice reference."
                )
            import torch
            from qwen_tts import Qwen3TTSModel

            device = qwen3_device
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
                    qwen3_model_id,
                    device,
                    dtype,
                )
                self._qwen3_model = await asyncio.to_thread(
                    Qwen3TTSModel.from_pretrained,
                    qwen3_model_id,
                    device_map="cuda:0" if device == "cuda" else "cpu",
                    dtype=dtype,
                    attn_implementation="sdpa",
                )
                self._qwen3_loaded_device = device

            prompt_cache_path = await asyncio.to_thread(
                _qwen3_voice_prompt_cache_path,
                qwen3_reference_audio,
                qwen3_model_id,
                qwen3_reference_text,
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
                x_vector_only = not bool(qwen3_reference_text.strip())
                prompt_items = await asyncio.to_thread(
                    self._qwen3_model.create_voice_clone_prompt,
                    ref_audio=qwen3_reference_audio,
                    ref_text=qwen3_reference_text or None,
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
                language=qwen_languages[language],
                speed=_QWEN3_SPEED[voice_speed],
                volume=_VOICE_VOLUMES[voice_volume][1],
            )
        return tts
