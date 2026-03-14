"""Direct OpenAI-compatible provider — bypasses LiteLLM."""

from __future__ import annotations

import uuid
from typing import Any

import json_repair
from openai import AsyncOpenAI

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from nanobot.providers.responses_api import (
    convert_messages as convert_responses_messages,
    convert_tools as convert_responses_tools,
    normalize_tool_choice as normalize_responses_tool_choice,
    parse_response as parse_responses_response,
    prompt_cache_key as responses_prompt_cache_key,
)


class CustomProvider(LLMProvider):

    def __init__(
        self,
        api_key: str = "no-key",
        api_base: str = "http://localhost:8000/v1",
        default_model: str = "default",
        api: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        super().__init__(api_key, api_base)
        self.default_model = default_model
        self.api = self._normalize_api(api)
        # Keep affinity stable for this provider instance to improve backend cache locality.
        headers = {"x-session-affinity": uuid.uuid4().hex}
        if extra_headers:
            headers.update(extra_headers)
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            default_headers=headers,
        )

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        if self.api == "openai-responses":
            return await self._chat_responses(
                messages=messages,
                tools=tools,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                tool_choice=tool_choice,
            )
        return await self._chat_completions(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )

    async def _chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": self._sanitize_empty_content(messages),
            "max_tokens": max(1, max_tokens),
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if tools:
            kwargs.update(tools=tools, tool_choice=tool_choice or "auto")
        try:
            return self._parse(await self._client.chat.completions.create(**kwargs))
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    async def _chat_responses(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        instructions, input_items = convert_responses_messages(self._sanitize_empty_content(messages))
        kwargs: dict[str, Any] = {
            "model": model or self.default_model,
            "input": input_items,
            "max_output_tokens": max(1, max_tokens),
            "store": False,
            "parallel_tool_calls": True,
            "prompt_cache_key": responses_prompt_cache_key(messages),
        }
        if instructions:
            kwargs["instructions"] = instructions
        if temperature is not None:
            kwargs["temperature"] = temperature
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        if tools:
            kwargs["tools"] = convert_responses_tools(tools)
            kwargs["tool_choice"] = normalize_responses_tool_choice(tool_choice or "auto")
        elif tool_choice not in (None, "auto"):
            kwargs["tool_choice"] = normalize_responses_tool_choice(tool_choice)
        try:
            response = await self._client.responses.create(**kwargs)
            return parse_responses_response(response)
        except Exception as e:
            return LLMResponse(content=f"Error: {e}", finish_reason="error")

    def _parse(self, response: Any) -> LLMResponse:
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCallRequest(id=tc.id, name=tc.function.name,
                            arguments=json_repair.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]
        u = response.usage
        return LLMResponse(
            content=msg.content, tool_calls=tool_calls, finish_reason=choice.finish_reason or "stop",
            usage={"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens} if u else {},
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
        )

    def get_default_model(self) -> str:
        return self.default_model

    @staticmethod
    def _normalize_api(api: str | None) -> str:
        value = (api or "chat-completions").strip().lower()
        if value in {"responses", "openai-responses", "openai_responses"}:
            return "openai-responses"
        return "chat-completions"
