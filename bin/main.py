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

    tray = SystemTrayController(_schedule_toggle_session, _request_shutdown)
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
