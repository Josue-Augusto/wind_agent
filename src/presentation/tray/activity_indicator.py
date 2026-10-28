from __future__ import annotations

import logging
from threading import Event, RLock, Thread
from typing import Callable

from PIL import Image, ImageDraw

_LOGGER = logging.getLogger(__name__)

_STATE_LABELS = {
    "idle": "pronto para iniciar",
    "starting": "iniciando sessão",
    "listening": "ouvindo",
    "capturing": "capturando sua fala",
    "processing": "processando resposta",
    "speaking": "falando",
}
_STATE_COLORS = {
    "idle": "#10B981",
    "starting": "#F59E0B",
    "listening": "#2563EB",
    "capturing": "#06B6D4",
    "processing": "#F59E0B",
    "speaking": "#8B5CF6",
}


def create_activity_icon_image(state: str = "idle", phase: int = 0) -> Image.Image:
    """Render the compact tray glyph for an assistant state."""
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((3, 3, 61, 61), fill="#111827")
    draw.ellipse((7, 7, 57, 57), fill=_STATE_COLORS.get(state, "#2563EB"))

    if state in {"capturing", "speaking"}:
        heights = (12, 20, 28, 20, 12)
        offsets = (0, 2, 4, 1, 3) if state == "speaking" else (3, 1, 4, 2, 0)
        for index, base_height in enumerate(heights):
            bar_height = max(
                8, base_height + (offsets[(index + phase) % 5] - 2) * 3
            )
            x = 17 + index * 7
            top = (64 - bar_height) // 2
            draw.rounded_rectangle(
                (x, top, x + 4, top + bar_height), radius=2, fill="white"
            )
    else:
        draw.rounded_rectangle((26, 12, 38, 39), radius=6, fill="white")
        draw.arc((18, 17, 46, 45), start=0, end=180, fill="white", width=4)
        draw.line((32, 45, 32, 52), fill="white", width=4)
        draw.line((24, 53, 40, 53), fill="white", width=4)
    return image


class TrayActivityIndicator:
    """Update a tray icon and tooltip as the voice pipeline changes state."""

    def __init__(
        self,
        set_icon: Callable[[Image.Image], None],
        set_title: Callable[[str], None],
    ) -> None:
        self._set_icon = set_icon
        self._set_title = set_title
        self._state = "idle"
        self._lock = RLock()
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(
            target=self._animate,
            name="tray-activity-animation",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)

    def set_state(self, state: str) -> None:
        label = _STATE_LABELS.get(state, state)
        with self._lock:
            changed = state != self._state
            self._state = state
            self._set_title(f"Assistente de voz — {label}")
            if changed:
                self._set_icon(create_activity_icon_image(state))

    def _animate(self) -> None:
        phase = 0
        while not self._stop_event.wait(0.18):
            with self._lock:
                if self._state not in {"capturing", "speaking"}:
                    continue
                state = self._state
                self._set_icon(create_activity_icon_image(state, phase))
            phase += 1
