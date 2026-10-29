from __future__ import annotations

import json
import logging
import random
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

_LOGGER = logging.getLogger(__name__)


class LocalLMStudioMCPService(LLMService):
    """Bridge Pipecat turns to LM Studio's native MCP-capable chat endpoint."""

    Settings = LLMSettings

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        mcp_servers: list[str],
        language: str = "pt",
    ) -> None:
        super().__init__(settings=self.Settings(model=model))
        self._endpoint = self._chat_endpoint(base_url)
        self._api_key = api_key
        self._integrations = [f"mcp/{server.strip()}" for server in mcp_servers]
        self._language = language
        self._previous_response_id: str | None = None

    @staticmethod
    def _chat_endpoint(base_url: str) -> str:
        parsed = urlsplit(base_url.rstrip("/"))
        path = parsed.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit((parsed.scheme, parsed.netloc, f"{path}/api/v1/chat", "", ""))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self._previous_response_id = None
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        try:
            text = await self._complete(frame.context)
            if text:
                spoken_text = self._to_spoken_text(text)
                if spoken_text:
                    await self.push_frame(LLMTextFrame(text=spoken_text))
        except Exception as error:
            _LOGGER.exception("LM Studio MCP request failed.")
            await self.push_error(
                error_msg=f"LM Studio MCP request failed: {error}",
                exception=error,
            )
        finally:
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    def _search_acknowledgement(self) -> str:
        phrases = {
            "en": (
                "Sure, let me check that for you. One moment.",
                "I’ll look into that now. Just a moment.",
                "Let me find out for you. Give me a second.",
            ),
            "es": (
                "Claro, déjame comprobarlo. Un momento.",
                "Voy a buscarlo ahora. Dame un instante.",
                "Déjame revisar eso por ti. Un segundo.",
            ),
        }.get(
            self._language,
            (
                "Certo, deixa eu conferir isso aqui. Um momento.",
                "Vou pesquisar isso para você. Só um instante.",
                "Deixa eu verificar essa informação. Já te respondo.",
                "Beleza, vou dar uma olhada nisso. Um segundinho.",
            ),
        )
        return random.choice(phrases)

    @staticmethod
    def _request_explicitly_asks_for_search(text: str) -> bool:
        return bool(
            re.search(
                r"\b(busca(?:r)?|pesquisa(?:r)?|procura(?:r)?|consulta(?:r)?|"
                r"investiga(?:r)?|na internet|na web|online|"
                r"search|look up|find out|research|check online|"
                r"busca|busca en internet|investiga|consulta)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _to_spoken_text(text: str) -> str:
        """Remove Markdown syntax that speech engines may fail to pronounce."""
        text = re.sub(r"```[^\n]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"[*_~]{1,3}", "", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    async def _complete(self, context: LLMContext) -> str:
        messages = self.get_llm_adapter().get_messages(context)
        system_prompt = "\n\n".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") in {"system", "developer"}
        ).strip()
        user_message = next(
            (message for message in reversed(messages) if message.get("role") == "user"),
            None,
        )
        if user_message is None:
            return ""
        user_input = user_message.get("content", "")
        if not isinstance(user_input, str):
            _LOGGER.warning("Skipping unsupported non-text LM Studio user input.")
            return ""

        announced_tool_call = self._request_explicitly_asks_for_search(user_input)
        if announced_tool_call:
            await self.push_frame(LLMTextFrame(text=self._search_acknowledgement()))

        payload: dict[str, Any] = {
            "model": self._settings.model,
            "input": user_input,
            "integrations": self._integrations,
            "store": True,
        }
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if self._previous_response_id:
            payload["previous_response_id"] = self._previous_response_id

        await self.start_ttfb_metrics()
        payload["stream"] = True
        result: dict[str, Any] | None = None
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream(
                "POST",
                self._endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.is_error:
                    detail = (await response.aread()).decode(
                        response.encoding or "utf-8", errors="replace"
                    ).strip().replace("\n", " ")[:500]
                    raise RuntimeError(
                        f"LM Studio returned HTTP {response.status_code}"
                        + (f": {detail}" if detail else ".")
                    )

                data_lines: list[str] = []
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                    elif not line and data_lines:
                        event_data = "\n".join(data_lines)
                        data_lines.clear()
                        if event_data:
                            event = self._parse_stream_event(event_data)
                            if (
                                event.get("type") == "tool_call.start"
                                and not announced_tool_call
                            ):
                                announced_tool_call = True
                                await self.push_frame(
                                    LLMTextFrame(text=self._search_acknowledgement())
                                )
                            elif event.get("type") == "chat.end":
                                candidate = event.get("result")
                                if isinstance(candidate, dict):
                                    result = candidate
                if data_lines:
                    event = self._parse_stream_event("\n".join(data_lines))
                    if (
                        event.get("type") == "tool_call.start"
                        and not announced_tool_call
                    ):
                        await self.push_frame(
                            LLMTextFrame(text=self._search_acknowledgement())
                        )
                    elif event.get("type") == "chat.end":
                        candidate = event.get("result")
                        if isinstance(candidate, dict):
                            result = candidate
        await self.stop_ttfb_metrics()
        if result is None:
            raise RuntimeError("LM Studio stream ended without a chat.end result.")
        if not isinstance(result, dict):
            raise RuntimeError("LM Studio returned an invalid chat response.")
        response_id = result.get("response_id")
        if isinstance(response_id, str) and response_id.startswith("resp_"):
            self._previous_response_id = response_id

        output = result.get("output", [])
        if not isinstance(output, list):
            raise RuntimeError("LM Studio returned an invalid output list.")
        messages_text = [
            text
            for item in output
            if isinstance(item, dict)
            and item.get("type") == "message"
            and (text := self._message_text(item.get("content")))
        ]
        return "\n".join(messages_text).strip()

    @staticmethod
    def _parse_stream_event(data: str) -> dict[str, Any]:
        try:
            event = json.loads(data)
        except ValueError as error:
            raise RuntimeError("LM Studio returned an invalid stream event.") from error
        if not isinstance(event, dict):
            raise RuntimeError("LM Studio returned an invalid stream event.")
        return event

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            return "".join(parts)
        return ""
