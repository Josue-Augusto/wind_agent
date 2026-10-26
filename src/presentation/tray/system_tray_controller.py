from collections.abc import Callable
from threading import RLock
from typing import Sequence

import pystray
from PIL import Image, ImageDraw


class SystemTrayController:
    def __init__(
        self,
        on_toggle_session: Callable[[], None],
        on_exit: Callable[[], None],
        audio_input_devices: Sequence[tuple[int, str]],
        selected_audio_input_device: int,
        on_select_audio_input_device: Callable[[int], None],
        conversation_languages: Sequence[tuple[str, str]],
        selected_conversation_language: str,
        voices_by_language: dict[str, Sequence[tuple[str, str]]],
        selected_voice: str,
        selected_voices_by_language: dict[str, str],
        on_select_conversation_language: Callable[[str], None],
        on_select_voice: Callable[[str], None],
        voice_speed_options: Sequence[tuple[str, str]],
        selected_voice_speed: str,
        on_select_voice_speed: Callable[[str], None],
        voice_style_options: Sequence[tuple[str, str]],
        selected_voice_style: str,
        on_select_voice_style: Callable[[str], None],
        voice_volume_options: Sequence[tuple[str, str]],
        selected_voice_volume: str,
        on_select_voice_volume: Callable[[str], None],
        llm_providers: Sequence[tuple[str, str]],
        selected_llm_provider: str,
        on_select_llm_provider: Callable[[str], None],
    ) -> None:
        self._llm_providers = tuple(llm_providers)
        self._selected_llm_provider = selected_llm_provider
        self._on_select_llm_provider = on_select_llm_provider
        self._audio_input_devices = tuple(audio_input_devices)
        self._selected_audio_input_device = selected_audio_input_device
        self._on_select_audio_input_device = on_select_audio_input_device
        self._conversation_languages = tuple(conversation_languages)
        self._selected_conversation_language = selected_conversation_language
        self._voices_by_language = {
            language: tuple(voices)
            for language, voices in voices_by_language.items()
        }
        self._selected_voice = selected_voice
        self._selected_voices_by_language = dict(selected_voices_by_language)
        self._on_select_conversation_language = on_select_conversation_language
        self._on_select_voice = on_select_voice
        self._voice_speed_options = tuple(voice_speed_options)
        self._selected_voice_speed = selected_voice_speed
        self._on_select_voice_speed = on_select_voice_speed
        self._voice_style_options = tuple(voice_style_options)
        self._selected_voice_style = selected_voice_style
        self._on_select_voice_style = on_select_voice_style
        self._voice_volume_options = tuple(voice_volume_options)
        self._selected_voice_volume = selected_voice_volume
        self._on_select_voice_volume = on_select_voice_volume
        self._icon = pystray.Icon(
            "wind_agent",
            self._create_icon_image(),
            "Assistente de voz",
            self._create_menu(),
        )
        self._on_toggle_session = on_toggle_session
        self._on_exit = on_exit
        self._lock = RLock()
        self._started = False

    def _create_menu(self) -> pystray.Menu:
        return pystray.Menu(
            pystray.MenuItem("Iniciar/parar sessão", self._handle_toggle),
            pystray.MenuItem(
                "Provedor de IA (próxima sessão)", self._create_llm_provider_menu()
            ),
            pystray.MenuItem(
                "Microfone (próxima sessão)", self._create_audio_input_menu()
            ),
            pystray.MenuItem(
                "Idioma da conversa (próxima sessão)",
                self._create_conversation_language_menu(),
            ),
            pystray.MenuItem(
                "Voz (próxima sessão)",
                self._create_voice_menu(),
            ),
            pystray.MenuItem(
                "Configuração da voz (próxima sessão)",
                pystray.Menu(
                    pystray.MenuItem(
                        "Velocidade", self._create_voice_speed_menu()
                    ),
                    pystray.MenuItem(
                        "Estilo", self._create_voice_style_menu()
                    ),
                    pystray.MenuItem(
                        "Volume", self._create_voice_volume_menu()
                    ),
                ),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Sair", self._handle_exit),
        )

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

    def _create_audio_input_menu(self) -> pystray.Menu:
        def make_action(device_index: int) -> Callable:
            def select_device(icon: pystray.Icon, item: pystray.MenuItem) -> None:
                self._selected_audio_input_device = device_index
                self._on_select_audio_input_device(device_index)
                icon.update_menu()

            return select_device

        items = [
            pystray.MenuItem(
                label,
                make_action(device_index),
                checked=lambda item, index=device_index: (
                    index == self._selected_audio_input_device
                ),
                radio=True,
            )
            for device_index, label in self._audio_input_devices
        ]
        if not items:
            items.append(
                pystray.MenuItem(
                    "Nenhum microfone disponível", None, enabled=False
                )
            )
        return pystray.Menu(*items)

    def _create_llm_provider_menu(self) -> pystray.Menu:
        def make_action(provider: str) -> Callable:
            def select_provider(icon: pystray.Icon, item: pystray.MenuItem) -> None:
                self._on_select_llm_provider(provider)
                self._selected_llm_provider = provider
                icon.update_menu()

            return select_provider

        if not self._llm_providers:
            return pystray.Menu(
                pystray.MenuItem("Nenhum provedor configurado", None, enabled=False)
            )
        return pystray.Menu(
            *(
                pystray.MenuItem(
                    label,
                    make_action(provider),
                    checked=lambda item, name=provider: (
                        name == self._selected_llm_provider
                    ),
                    radio=True,
                )
                for provider, label in self._llm_providers
            )
        )

    def _create_conversation_language_menu(self) -> pystray.Menu:
        def make_action(language: str) -> Callable:
            def select_language(icon: pystray.Icon, item: pystray.MenuItem) -> None:
                self._selected_conversation_language = language
                voices = self._voices_by_language.get(language, ())
                if voices:
                    self._selected_voice = self._selected_voices_by_language.get(
                        language, voices[0][0]
                    )
                self._on_select_conversation_language(language)
                icon.menu = self._create_menu()
                icon.update_menu()

            return select_language

        return pystray.Menu(
            *(
                pystray.MenuItem(
                    label,
                    make_action(language),
                    checked=lambda item, code=language: (
                        code == self._selected_conversation_language
                    ),
                    radio=True,
                )
                for language, label in self._conversation_languages
            )
        )

    def _create_voice_menu(self) -> pystray.Menu:
        voices = self._voices_by_language.get(
            self._selected_conversation_language, ()
        )

        def make_action(voice: str) -> Callable:
            def select_voice(icon: pystray.Icon, item: pystray.MenuItem) -> None:
                self._selected_voice = voice
                self._selected_voices_by_language[
                    self._selected_conversation_language
                ] = voice
                self._on_select_voice(voice)
                icon.update_menu()

            return select_voice

        return pystray.Menu(
            *(
                pystray.MenuItem(
                    label,
                    make_action(voice),
                    checked=lambda item, voice_id=voice: (
                        voice_id == self._selected_voice
                    ),
                    radio=True,
                )
                for voice, label in voices
            )
        )

    def _create_voice_speed_menu(self) -> pystray.Menu:
        return self._create_voice_option_menu(
            self._voice_speed_options,
            "_selected_voice_speed",
            self._on_select_voice_speed,
        )

    def _create_voice_style_menu(self) -> pystray.Menu:
        return self._create_voice_option_menu(
            self._voice_style_options,
            "_selected_voice_style",
            self._on_select_voice_style,
        )

    def _create_voice_volume_menu(self) -> pystray.Menu:
        return self._create_voice_option_menu(
            self._voice_volume_options,
            "_selected_voice_volume",
            self._on_select_voice_volume,
        )

    def _create_voice_option_menu(
        self,
        options: Sequence[tuple[str, str]],
        selected_attribute: str,
        on_select: Callable[[str], None],
    ) -> pystray.Menu:
        def make_action(value: str) -> Callable:
            def select_option(icon: pystray.Icon, item: pystray.MenuItem) -> None:
                setattr(self, selected_attribute, value)
                on_select(value)
                icon.update_menu()

            return select_option

        return pystray.Menu(
            *(
                pystray.MenuItem(
                    label,
                    make_action(value),
                    checked=lambda item, option=value: (
                        option == getattr(self, selected_attribute)
                    ),
                    radio=True,
                )
                for value, label in options
            )
        )

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
