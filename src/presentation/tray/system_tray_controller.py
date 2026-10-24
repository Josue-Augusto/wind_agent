from collections.abc import Callable
from threading import RLock

import pystray
from PIL import Image, ImageDraw


class SystemTrayController:
    def __init__(
        self,
        on_toggle_session: Callable[[], None],
        on_exit: Callable[[], None],
    ) -> None:
        menu = pystray.Menu(
            pystray.MenuItem("Iniciar/parar sessão", self._handle_toggle),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Sair", self._handle_exit),
        )
        self._icon = pystray.Icon(
            "wind_agent",
            self._create_icon_image(),
            "Assistente de voz",
            menu,
        )
        self._on_toggle_session = on_toggle_session
        self._on_exit = on_exit
        self._lock = RLock()
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("The system tray is already running.")
            self._started = True

        try:
            self._icon.run_detached()
        except Exception:
            with self._lock:
                self._started = False
            raise

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            try:
                self._icon.stop()
            finally:
                self._started = False

    def set_agent_state(self, state: str) -> None:
        state_labels = {
            "idle": "inativo",
            "listening": "ouvindo",
            "processing": "carregando/processando",
            "speaking": "falando",
        }
        label = state_labels.get(state, state)
        self._icon.title = f"Assistente de voz — {label}"

    def _handle_toggle(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._on_toggle_session()

    def _handle_exit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        try:
            self._on_exit()
        finally:
            self.stop()

    @staticmethod
    def _create_icon_image() -> Image.Image:
        image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.ellipse((3, 3, 61, 61), fill="#2563EB")
        draw.rounded_rectangle((26, 12, 38, 39), radius=6, fill="white")
        draw.arc((18, 17, 46, 45), start=0, end=180, fill="white", width=4)
        draw.line((32, 45, 32, 52), fill="white", width=4)
        draw.line((24, 53, 40, 53), fill="white", width=4)
        return image
