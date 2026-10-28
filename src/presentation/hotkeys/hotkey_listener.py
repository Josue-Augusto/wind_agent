from collections.abc import Callable
from threading import RLock, current_thread

from pynput import keyboard
from pynput.keyboard import Key


class HotkeyListener:
    def __init__(
        self,
        on_activate: Callable[[], None],
        is_push_to_talk_enabled: Callable[[], bool],
        on_push_to_talk: Callable[[bool], None],
        is_session_active: Callable[[], bool],
    ) -> None:
        self._on_activate = on_activate
        self._is_push_to_talk_enabled = is_push_to_talk_enabled
        self._on_push_to_talk = on_push_to_talk
        self._is_session_active = is_session_active
        self._listener: keyboard.Listener | None = None
        self._pressed_keys: set[Key] = set()
        self._combo_active = False
        self._lock = RLock()

    def start(self) -> None:
        if self._listener is not None and self._listener.is_alive():
            raise RuntimeError("The global hotkey listener is already running.")

        listener = keyboard.Listener(
            on_press=self._handle_key_press,
            on_release=self._handle_key_release,
        )
        listener.start()
        self._listener = listener

    def stop(self) -> None:
        listener, self._listener = self._listener, None
        if listener is None:
            return

        listener.stop()
        if current_thread() is not listener:
            listener.join()

    def _handle_key_press(self, key: Key) -> None:
        with self._lock:
            self._pressed_keys.add(key)
            shift_pressed = any(
                item in self._pressed_keys
                for item in (Key.shift, Key.shift_l, Key.shift_r)
            )
            if not shift_pressed or Key.space not in self._pressed_keys:
                return
            if self._combo_active:
                return
            self._combo_active = True

        if self._is_push_to_talk_enabled():
            self._on_push_to_talk(True)
            if not self._is_session_active():
                self._on_activate()
        else:
            self._on_activate()

    def _handle_key_release(self, key: Key) -> None:
        with self._lock:
            self._pressed_keys.discard(key)
            if not self._combo_active:
                return
            shift_pressed = any(
                item in self._pressed_keys
                for item in (Key.shift, Key.shift_l, Key.shift_r)
            )
            if shift_pressed and Key.space in self._pressed_keys:
                return
            self._combo_active = False

        if self._is_push_to_talk_enabled():
            self._on_push_to_talk(False)
