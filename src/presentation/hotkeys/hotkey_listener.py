from collections.abc import Callable
from threading import current_thread

from pynput.keyboard import GlobalHotKeys


class HotkeyListener:
    HOTKEY = "<shift>+<space>"

    def __init__(self, on_activate: Callable[[], None]) -> None:
        self._on_activate = on_activate
        self._listener: GlobalHotKeys | None = None

    def start(self) -> None:
        if self._listener is not None and self._listener.is_alive():
            raise RuntimeError("The global hotkey listener is already running.")

        listener = GlobalHotKeys({self.HOTKEY: self._handle_activate})
        listener.start()
        self._listener = listener

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is None:
            return

        listener.stop()
        if current_thread() is not listener:
            listener.join()

    def _handle_activate(self) -> None:
        self._on_activate()
