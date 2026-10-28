from __future__ import annotations

import logging
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
    ) -> None:
        super().__init__(settings=self.Settings(model=model))
        self._endpoint = self._chat_endpoint(base_url)
        self._api_key = api_key
        self._integrations = [f"mcp/{server.strip()}" for server in mcp_servers]
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
                await self.push_frame(LLMTextFrame(text=text))
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
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                self._endpoint,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            if response.is_error:
                detail = response.text.strip().replace("\n", " ")[:500]
                raise RuntimeError(
                    f"LM Studio returned HTTP {response.status_code}"
                    + (f": {detail}" if detail else ".")
                )
        await self.stop_ttfb_metrics()
        result = response.json()
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
