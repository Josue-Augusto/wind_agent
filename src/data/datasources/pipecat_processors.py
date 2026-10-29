from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from concurrent.futures import Future
from typing import Any, Callable

import sounddevice as sd

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
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
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts
from src.domain.entities.agent_state import AgentState

_LOGGER = logging.getLogger(__name__)

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
        self._push_to_talk_enabled = False
        self._push_to_talk_pressed = False
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

    def set_push_to_talk(self, *, enabled: bool, pressed: bool = False) -> None:
        self._push_to_talk_enabled = enabled
        self._push_to_talk_pressed = pressed

    def set_push_to_talk_pressed(self, pressed: bool) -> None:
        self._push_to_talk_pressed = pressed

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

        if self._push_to_talk_enabled and not self._push_to_talk_pressed:
            audio = audio * 0
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
        if isinstance(frame, InterruptionFrame):
            self._abort_audio()
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM and isinstance(
            frame, TTSAudioRawFrame
        ):
            chunk_size = max(1, frame.sample_rate // 50) * frame.num_channels * 2
            for offset in range(0, len(frame.audio), chunk_size):
                write_task = asyncio.create_task(
                    asyncio.to_thread(
                        self._write_audio_chunk,
                        frame,
                        offset,
                        offset + chunk_size,
                    )
                )
                try:
                    await asyncio.shield(write_task)
                except asyncio.CancelledError:
                    # Cancelling to_thread() does not stop its worker thread.
                    # Finish this short write before the interruption closes
                    # the PortAudio stream it is using.
                    await asyncio.shield(write_task)
                    raise
                if not write_task.result():
                    break
        await self.push_frame(frame, direction)

    def _write_audio_chunk(
        self, frame: TTSAudioRawFrame, start: int, end: int
    ) -> bool:
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

        try:
            self._stream.write(frame.audio[start:end])
        except sd.PortAudioError:
            _LOGGER.warning(
                "Audio output failed; closing the output stream.", exc_info=True
            )
            try:
                self._close_stream()
            except sd.PortAudioError:
                _LOGGER.debug("Could not close the failed audio stream.", exc_info=True)
            return False
        return True

    def _abort_audio(self) -> None:
        """Discard queued playback immediately when the user starts speaking."""
        stream, self._stream = self._stream, None
        self._stream_format = None
        if stream is None:
            return
        try:
            stream.abort()
        except Exception:
            _LOGGER.debug("Could not abort interrupted audio playback.", exc_info=True)
        try:
            stream.close()
        except Exception:
            _LOGGER.debug("Could not close interrupted audio playback.", exc_info=True)

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
                self._on_state_change(AgentState.CAPTURING)
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                _LOGGER.info("VAD detected the end of the utterance; Whisper will transcribe it.")
                self._on_state_change(AgentState.PROCESSING)
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


