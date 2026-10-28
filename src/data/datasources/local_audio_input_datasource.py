from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import sounddevice as sd

from src.data.datasources.local_pipecat_config import (
    LocalPipecatSettings,
    _USER_SETTINGS_FILE,
)

_LOGGER = logging.getLogger(__name__)


class LocalAudioInputDataSource:
    """Discover, select, and persist the microphone used by Pipecat sessions."""

    def __init__(self, settings: LocalPipecatSettings) -> None:
        self._settings = settings
        self._audio_devices = self._discover_audio_input_devices()
        self._audio_input_device = self._resolve_initial_audio_input_device()

    def _discover_audio_input_devices(self) -> list[dict[str, object]]:
        host_apis = sd.query_hostapis()
        devices = sd.query_devices()
        candidates = [
            (index, device, host_apis[device["hostapi"]]["name"])
            for index, device in enumerate(devices)
            if device["max_input_channels"] >= self._settings.audio_channels
            and str(device["name"]).strip()
            and not any(
                marker in str(device["name"]).casefold()
                for marker in (
                    "mapper",
                    "mapeador",
                    "driver de captura de som primário",
                    "primary sound driver",
                )
            )
        ]
        unique_devices: list[dict[str, object]] = []
        grouped_devices: dict[str, list[tuple[int, Any, str]]] = {}
        for candidate in candidates:
            normalized_name = str(candidate[1]["name"]).strip().casefold()
            grouped_devices.setdefault(normalized_name, []).append(candidate)

        api_priority = {
            "MME": 0,
            "Windows DirectSound": 1,
            "Windows WASAPI": 2,
            "Windows WDM-KS": 3,
        }
        for name_candidates in grouped_devices.values():
            compatible_candidates = []
            for candidate in name_candidates:
                index, device, host_api_name = candidate
                try:
                    sd.check_input_settings(
                        device=index,
                        channels=self._settings.audio_channels,
                        dtype="int16",
                        samplerate=self._settings.audio_sample_rate,
                    )
                except sd.PortAudioError:
                    continue
                capture_channels = 1
                if device["max_input_channels"] >= 2:
                    try:
                        sd.check_input_settings(
                            device=index,
                            channels=2,
                            dtype="int16",
                            samplerate=self._settings.audio_sample_rate,
                        )
                        capture_channels = 2
                    except sd.PortAudioError:
                        pass
                compatible_candidates.append(
                    (*candidate, capture_channels)
                )
            if not compatible_candidates:
                continue
            index, device, host_api_name, capture_channels = min(
                compatible_candidates,
                key=lambda candidate: api_priority.get(candidate[2], 99),
            )
            unique_devices.append(
                {
                    "index": index,
                    "name": str(device["name"]).strip(),
                    "host_api": host_api_name,
                    "capture_channels": capture_channels,
                }
            )
        return unique_devices

    def get_audio_input_devices(self) -> list[tuple[int, str]]:
        return [
            (int(device["index"]), str(device["name"]))
            for device in self._audio_devices
        ]

    def get_selected_audio_input_device(self) -> int:
        return self._audio_input_device

    def get_capture_channels(self) -> int:
        device = next(
            item
            for item in self._audio_devices
            if item["index"] == self._audio_input_device
        )
        return int(device["capture_channels"])

    def select_audio_input_device(self, device_index: int) -> None:
        selected_device = next(
            (device for device in self._audio_devices if device["index"] == device_index),
            None,
        )
        if selected_device is None:
            raise ValueError(f"Audio input device {device_index} is unavailable.")
        self._audio_input_device = device_index
        self._save_user_settings(
            {
                "audio_input_device": {
                    "name": selected_device["name"],
                    "host_api": selected_device["host_api"],
                }
            }
        )

    def _resolve_initial_audio_input_device(self) -> int:
        requested = self._settings.audio_input_device
        if requested is not None:
            try:
                raw_device = sd.query_devices(requested, "input")
                match = self._find_audio_device(str(raw_device["name"]))
                if match is not None:
                    return match
            except (ValueError, sd.PortAudioError):
                _LOGGER.warning(
                    "Configured audio input device %r is unavailable.", requested
                )

        saved_device = self._load_saved_audio_device()
        if saved_device is not None:
            match = next(
                (
                    int(device["index"])
                    for device in self._audio_devices
                    if device["name"] == saved_device.get("name")
                ),
                None,
            )
            if match is not None:
                _LOGGER.info(
                    "Restored saved audio input device: %s", saved_device["name"]
                )
                return match

        try:
            default_device = sd.query_devices(kind="input")
            match = self._find_audio_device(str(default_device["name"]))
            if match is not None:
                _LOGGER.info(
                    "Using the Windows default audio input: %s",
                    default_device["name"],
                )
                return match
        except (ValueError, sd.PortAudioError):
            pass

        if not self._audio_devices:
            raise RuntimeError("No active audio input devices were found.")
        fallback = int(self._audio_devices[0]["index"])
        _LOGGER.warning(
            "Windows default input was unavailable; using %s.",
            self._audio_devices[0]["name"],
        )
        return fallback

    def _find_audio_device(self, name: str) -> int | None:
        for device in self._audio_devices:
            if str(device["name"]).casefold() == name.casefold():
                return int(device["index"])
        return None

    @staticmethod
    def _load_saved_audio_settings() -> dict[str, Any]:
        try:
            settings = json.loads(_USER_SETTINGS_FILE.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        return settings if isinstance(settings, dict) else {}

    @staticmethod
    def _load_saved_audio_device() -> dict[str, str] | None:
        audio_device = LocalAudioInputDataSource._load_saved_audio_settings().get(
            "audio_input_device"
        )
        if (
            isinstance(audio_device, dict)
            and isinstance(audio_device.get("name"), str)
            and isinstance(audio_device.get("host_api"), str)
        ):
            return audio_device
        return None

    @staticmethod
    def _save_user_settings(updates: dict[str, Any]) -> None:
        settings = LocalAudioInputDataSource._load_saved_audio_settings()
        settings.update(updates)
        try:
            _USER_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            temporary_file = _USER_SETTINGS_FILE.with_suffix(".tmp")
            temporary_file.write_text(
                json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary_file.replace(_USER_SETTINGS_FILE)
        except OSError:
            _LOGGER.exception("Could not save user settings.")

