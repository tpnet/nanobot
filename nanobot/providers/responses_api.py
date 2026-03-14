"""Shared helpers for OpenAI Responses API providers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import json_repair

from nanobot.providers.base import LLMResponse, ToolCallRequest

_FINISH_REASON_MAP = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "error",
    "cancelled": "error",
}


def convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI chat-completions function tools to Responses API tools."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        converted.append(
            {
                "type": "function",
                "name": name,
                "description": fn.get("description") or "",
                "parameters": params if isinstance(params, dict) else {},
            }
        )
    return converted


def convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Convert chat-completions style messages into Responses API input items."""
    system_prompt = ""
    input_items: list[dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            system_prompt = content if isinstance(content, str) else ""
            continue

        if role == "user":
            input_items.append(_convert_user_message(content))
            continue

        if role == "assistant":
            if isinstance(content, str) and content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                        "status": "completed",
                        "id": f"msg_{idx}",
                    }
                )

            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                call_id, item_id = split_tool_call_id(tool_call.get("id"))
                call_id = call_id or f"call_{idx}"
                item_id = item_id or f"fc_{idx}"
                input_items.append(
                    {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue

        if role == "tool":
            call_id, _ = split_tool_call_id(msg.get("tool_call_id"))
            output_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
            )
            continue

    return system_prompt, input_items


def split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    """Split a stored tool_call_id into Responses API call_id and item_id."""
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None


def prompt_cache_key(messages: list[dict[str, Any]]) -> str:
    """Build a stable prompt cache key from serialized messages."""
    raw = json.dumps(messages, ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_tool_choice(tool_choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    """Convert chat-completions style tool_choice into Responses API shape."""
    if tool_choice is None or isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return tool_choice

    if tool_choice.get("type") == "function":
        fn = tool_choice.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return {"type": "function", "name": fn["name"]}
        if tool_choice.get("name"):
            return {"type": "function", "name": tool_choice["name"]}

    return tool_choice


def map_finish_reason(status: str | None) -> str:
    """Map Responses API status to the nanobot finish_reason vocabulary."""
    return _FINISH_REASON_MAP.get(status or "completed", "stop")


def parse_response(response: Any) -> LLMResponse:
    """Parse a non-streaming Responses API result into nanobot's provider shape."""
    error = getattr(response, "error", None)
    if error is not None:
        message = getattr(error, "message", None) or str(error)
        return LLMResponse(content=f"Error: {message}", finish_reason="error")

    tool_calls: list[ToolCallRequest] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "function_call":
            continue
        call_id = getattr(item, "call_id", None) or "call_0"
        item_id = getattr(item, "id", None)
        raw_args = getattr(item, "arguments", None) or "{}"
        try:
            args = json_repair.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            args = {"raw": raw_args}
        packed_id = f"{call_id}|{item_id}" if item_id else call_id
        tool_calls.append(
            ToolCallRequest(
                id=packed_id,
                name=getattr(item, "name", None) or "unknown_tool",
                arguments=args if isinstance(args, dict) else {"value": args},
            )
        )

    usage_data: dict[str, int] = {}
    usage = getattr(response, "usage", None)
    if usage is not None:
        usage_data = {
            "prompt_tokens": getattr(usage, "input_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "output_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

    status = getattr(response, "status", None)
    if status in {"failed", "cancelled"}:
        return LLMResponse(
            content="Error: Responses API request failed",
            finish_reason="error",
            usage=usage_data,
        )

    text = getattr(response, "output_text", None) or None
    finish_reason = "tool_calls" if tool_calls else map_finish_reason(status)
    return LLMResponse(
        content=text,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage_data,
    )


def _convert_user_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}
    if isinstance(content, list):
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append({"type": "input_image", "image_url": url, "detail": "auto"})
        if converted:
            return {"role": "user", "content": converted}
    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}
