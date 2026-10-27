import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from loguru import logger as pipecat_logger

from src.data.datasources.local_pipecat_datasource import LocalPipecatDataSource
from src.data.repositories.agent_repository_impl import AgentRepositoryImpl
from src.domain.entities.agent_state import AgentState
from src.domain.usecases.execute_voice_session import ExecuteVoiceSession
from src.presentation.hotkeys.hotkey_listener import HotkeyListener
from src.presentation.tray.system_tray_controller import SystemTrayController


_LOGGER = logging.getLogger(__name__)


def _choose_qwen3_reference_file() -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    window = tk.Tk()
    window.withdraw()
    window.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilename(
            parent=window,
            title="Selecione uma gravação WAV para a voz do Qwen3-TTS",
            filetypes=[("Áudio WAV", "*.wav"), ("Todos os arquivos", "*.*")],
        )
    finally:
        window.destroy()
    return selected or None


def _configure_logging() -> None:
    local_app_data = Path(
        os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))
    )
    log_directory = local_app_data / "WindAgent" / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            RotatingFileHandler(
                log_directory / "application.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            ),
            *([logging.StreamHandler(sys.stderr)] if sys.stderr is not None else []),
        ],
        force=True,
    )
    pipecat_logger.remove()
    pipecat_logger.add(
        log_directory / "pipecat.log",
        level="INFO",
        rotation="5 MB",
        retention=3,
        encoding="utf-8",
    )


async def _run_application() -> None:
    loop = asyncio.get_running_loop()
    shutdown_requested = asyncio.Event()
    pending_tasks: set[asyncio.Task[AgentState]] = set()
    shutting_down = False
    tray: SystemTrayController | None = None

    datasource = LocalPipecatDataSource()

    def _show_agent_state(state: AgentState) -> None:
        _LOGGER.info("Estado do assistente: %s", state.value)
        if tray is not None:
            try:
                loop.call_soon_threadsafe(tray.set_agent_state, state.value)
            except RuntimeError:
                _LOGGER.debug("Could not update tray status after loop shutdown.")

    repository = AgentRepositoryImpl(datasource, on_state_change=_show_agent_state)
    execute_voice_session = ExecuteVoiceSession(repository)

    def _report_task_result(task: asyncio.Task[AgentState]) -> None:
        pending_tasks.discard(task)
        if task.cancelled():
            return

        error = task.exception()
        if error is not None:
            _LOGGER.error(
                "The voice-session action failed.",
                exc_info=(type(error), error, error.__traceback__),
            )
            return

        _LOGGER.info("Agent state: %s", task.result().value)

    def _create_toggle_task() -> None:
        if shutting_down:
            return
        if pending_tasks:
            _LOGGER.info(
                "Ignoring session toggle while the previous session action is still in progress."
            )
            return
        task = asyncio.create_task(
            execute_voice_session.execute(),
            name="toggle-voice-session",
        )
        pending_tasks.add(task)
        task.add_done_callback(_report_task_result)

    def _schedule_toggle_session() -> None:
        try:
            _LOGGER.info("Shift+Space or tray toggle received.")
            loop.call_soon_threadsafe(_create_toggle_task)
        except RuntimeError as error:
            _LOGGER.warning("Could not schedule voice-session action: %s", error)

    def _request_shutdown() -> None:
        try:
            loop.call_soon_threadsafe(shutdown_requested.set)
        except RuntimeError as error:
            _LOGGER.warning("Could not schedule application shutdown: %s", error)

    def _select_audio_input_device(device_index: int) -> None:
        try:
            datasource.select_audio_input_device(device_index)
            _LOGGER.info(
                "Audio input device %d selected and saved for future sessions.",
                device_index,
            )
        except (RuntimeError, ValueError) as error:
            _LOGGER.error("Could not select audio input device %d: %s", device_index, error)

    def _schedule_audio_input_device_selection(device_index: int) -> None:
        try:
            loop.call_soon_threadsafe(_select_audio_input_device, device_index)
        except RuntimeError as error:
            _LOGGER.warning("Could not schedule audio device selection: %s", error)

    def _select_conversation_language(language: str) -> None:
        try:
            datasource.select_conversation_language(language)
            _LOGGER.info(
                "Conversation language set to %s for future sessions.", language
            )
        except ValueError as error:
            _LOGGER.error(
                "Could not select conversation language %s: %s", language, error
            )

    def _schedule_conversation_language_selection(language: str) -> None:
        try:
            loop.call_soon_threadsafe(_select_conversation_language, language)
        except RuntimeError as error:
            _LOGGER.warning("Could not schedule language selection: %s", error)

    def _select_voice(voice: str) -> None:
        try:
            datasource.select_voice(voice)
            _LOGGER.info("TTS voice %s selected for future sessions.", voice)
        except ValueError as error:
            _LOGGER.error("Could not select TTS voice %s: %s", voice, error)

    def _schedule_voice_selection(voice: str) -> None:
        try:
            loop.call_soon_threadsafe(_select_voice, voice)
        except RuntimeError as error:
            _LOGGER.warning("Could not schedule voice selection: %s", error)

    def _select_voice_speed(speed: str) -> None:
        try:
            datasource.select_voice_speed(speed)
            _LOGGER.info("Voice speed set to %s for future sessions.", speed)
        except ValueError as error:
            _LOGGER.error("Could not select voice speed %s: %s", speed, error)

    def _schedule_voice_speed_selection(speed: str) -> None:
        try:
            loop.call_soon_threadsafe(_select_voice_speed, speed)
        except RuntimeError as error:
            _LOGGER.warning("Could not schedule voice speed selection: %s", error)

    def _select_voice_style(style: str) -> None:
        try:
            datasource.select_voice_style(style)
            _LOGGER.info("Voice style set to %s for future sessions.", style)
        except ValueError as error:
            _LOGGER.error("Could not select voice style %s: %s", style, error)

    def _schedule_voice_style_selection(style: str) -> None:
        try:
            loop.call_soon_threadsafe(_select_voice_style, style)
        except RuntimeError as error:
            _LOGGER.warning("Could not schedule voice style selection: %s", error)

    def _select_voice_volume(volume: str) -> None:
        try:
            datasource.select_voice_volume(volume)
            _LOGGER.info("Voice volume set to %s for future sessions.", volume)
        except ValueError as error:
            _LOGGER.error("Could not select voice volume %s: %s", volume, error)

    def _schedule_voice_volume_selection(volume: str) -> None:
        try:
            loop.call_soon_threadsafe(_select_voice_volume, volume)
        except RuntimeError as error:
            _LOGGER.warning("Could not schedule voice volume selection: %s", error)

    def _select_llm_provider(provider: str) -> None:
        try:
            datasource.select_llm_provider(provider)
            _LOGGER.info("LLM provider %s selected for future sessions.", provider)
        except (OSError, ValueError) as error:
            _LOGGER.error("Could not select LLM provider %s: %s", provider, error)

    def _schedule_llm_provider_selection(provider: str) -> None:
        try:
            loop.call_soon_threadsafe(_select_llm_provider, provider)
        except RuntimeError as error:
            _LOGGER.warning("Could not schedule LLM provider selection: %s", error)

    def _select_tts_engine(engine: str) -> None:
        try:
            datasource.select_tts_engine(engine)
            if tray is not None:
                tray.update_voice_options(
                    datasource.get_voices_by_language(),
                    datasource.get_selected_voices_by_language(),
                )
            _LOGGER.info("TTS engine %s selected for future sessions.", engine)
        except (OSError, ValueError) as error:
            _LOGGER.error("Could not select TTS engine %s: %s", engine, error)

    def _schedule_tts_engine_selection(engine: str) -> None:
        try:
            loop.call_soon_threadsafe(_select_tts_engine, engine)
        except RuntimeError as error:
            _LOGGER.warning("Could not schedule TTS engine selection: %s", error)

    def _select_kokoro_voice_setting(setting: str, value: str) -> None:
        try:
            datasource.select_kokoro_voice_setting(setting, value)
            _LOGGER.info(
                "Kokoro voice setting %s set to %s for future sessions.",
                setting,
                value,
            )
        except (OSError, ValueError) as error:
            _LOGGER.error("Could not set Kokoro voice setting %s: %s", setting, error)

    def _schedule_kokoro_voice_setting(setting: str, value: str) -> None:
        try:
            loop.call_soon_threadsafe(
                _select_kokoro_voice_setting, setting, value
            )
        except RuntimeError as error:
            _LOGGER.warning("Could not schedule Kokoro voice setting: %s", error)

    async def _add_qwen3_reference() -> None:
        try:
            reference_path = await asyncio.to_thread(_choose_qwen3_reference_file)
            if not reference_path:
                return
            datasource.set_qwen3_reference_audio(reference_path)
            _LOGGER.info("Updated Qwen3-TTS voice reference: %s", reference_path)
        except Exception as error:
            _LOGGER.error("Could not update Qwen3-TTS voice reference: %s", error)

    def _schedule_add_qwen3_reference() -> None:
        try:
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(_add_qwen3_reference())
            )
        except RuntimeError as error:
            _LOGGER.warning("Could not open Qwen3-TTS reference picker: %s", error)

    tray = SystemTrayController(
        _schedule_toggle_session,
        _request_shutdown,
        datasource.get_audio_input_devices(),
        datasource.get_selected_audio_input_device(),
        _schedule_audio_input_device_selection,
        datasource.get_conversation_languages(),
        datasource.get_selected_conversation_language(),
        datasource.get_voices_by_language(),
        datasource.get_selected_voice(),
        datasource.get_selected_voices_by_language(),
        _schedule_conversation_language_selection,
        _schedule_voice_selection,
        datasource.get_voice_speed_options(),
        datasource.get_voice_adjustments()["speed"],
        _schedule_voice_speed_selection,
        datasource.get_voice_style_options(),
        datasource.get_voice_adjustments()["style"],
        _schedule_voice_style_selection,
        datasource.get_voice_volume_options(),
        datasource.get_voice_adjustments()["volume"],
        _schedule_voice_volume_selection,
        datasource.get_llm_providers(),
        datasource.get_selected_llm_provider(),
        _schedule_llm_provider_selection,
        datasource.get_tts_engines(),
        datasource.get_selected_tts_engine(),
        _schedule_tts_engine_selection,
        _schedule_add_qwen3_reference,
        datasource.get_kokoro_voice_options(),
        datasource.get_kokoro_voice_settings(),
        _schedule_kokoro_voice_setting,
    )
    hotkey = HotkeyListener(_schedule_toggle_session)
    tray.set_agent_state(repository.get_state().value)

    try:
        tray.start()
        hotkey.start()
        _LOGGER.info(
            "Voice assistant started; press Shift+Space to toggle a session."
        )
        await shutdown_requested.wait()
    finally:
        shutting_down = True
        try:
            await asyncio.to_thread(hotkey.stop)
        finally:
            try:
                await asyncio.to_thread(tray.stop)
            finally:
                try:
                    if pending_tasks:
                        await asyncio.gather(*tuple(pending_tasks))
                finally:
                    await repository.stop_session()


def main() -> None:
    _configure_logging()
    try:
        asyncio.run(_run_application())
    except KeyboardInterrupt:
        _LOGGER.info("Shutdown requested from the console.")
    except Exception:
        _LOGGER.exception("The voice assistant terminated unexpectedly.")
        raise


if __name__ == "__main__":
    main()
