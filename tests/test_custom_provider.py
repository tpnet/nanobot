from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.cli.commands import _make_provider
from nanobot.config.schema import Config
from nanobot.providers.custom_provider import CustomProvider


def test_config_accepts_custom_api_mode():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "custom", "model": "gpt-5.4"}},
            "providers": {
                "custom": {
                    "apiKey": "test-key",
                    "apiBase": "https://example.com/v1",
                    "api": "openai-responses",
                }
            },
        }
    )

    assert config.providers.custom.api == "openai-responses"


def test_make_provider_passes_custom_api_and_headers():
    config = Config.model_validate(
        {
            "agents": {"defaults": {"provider": "custom", "model": "gpt-5.4"}},
            "providers": {
                "custom": {
                    "apiKey": "test-key",
                    "apiBase": "https://example.com/v1",
                    "api": "openai-responses",
                    "extraHeaders": {"X-Test": "1"},
                }
            },
        }
    )

    provider = MagicMock()

    with patch("nanobot.providers.custom_provider.CustomProvider", return_value=provider) as custom_cls:
        created = _make_provider(config)

    custom_cls.assert_called_once_with(
        api_key="test-key",
        api_base="https://example.com/v1",
        default_model="gpt-5.4",
        api="openai-responses",
        extra_headers={"X-Test": "1"},
    )
    assert created is provider


@pytest.mark.asyncio
async def test_custom_provider_responses_mode_converts_payload():
    provider = CustomProvider(
        api_key="test-key",
        api_base="https://example.com/v1",
        default_model="gpt-5.4",
        api="openai-responses",
    )
    fake_response = SimpleNamespace(
        error=None,
        output=[],
        output_text="done",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7, total_tokens=18),
        status="completed",
    )
    create = AsyncMock(return_value=fake_response)
    provider._client = SimpleNamespace(
        responses=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())),
    )

    messages = [
        {"role": "system", "content": "You are nanobot."},
        {"role": "user", "content": "今天广州天气怎么样"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123|fc_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"广州"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123|fc_123",
            "name": "get_weather",
            "content": "广州，多云，28C",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    result = await provider.chat(
        messages=messages,
        tools=tools,
        max_tokens=321,
        temperature=0.2,
        reasoning_effort="high",
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )

    assert result.content == "done"
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "gpt-5.4"
    assert kwargs["instructions"] == "You are nanobot."
    assert kwargs["max_output_tokens"] == 321
    assert kwargs["temperature"] == 0.2
    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["tool_choice"] == {"type": "function", "name": "get_weather"}
    assert kwargs["tools"] == [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    assert kwargs["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "今天广州天气怎么样"}]},
        {
            "type": "function_call",
            "id": "fc_123",
            "call_id": "call_123",
            "name": "get_weather",
            "arguments": '{"city":"广州"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "广州，多云，28C",
        },
    ]


@pytest.mark.asyncio
async def test_custom_provider_responses_mode_parses_tool_calls():
    provider = CustomProvider(
        api_key="test-key",
        api_base="https://example.com/v1",
        default_model="gpt-5.4",
        api="responses",
    )
    fake_response = SimpleNamespace(
        error=None,
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_weather",
                id="fc_weather",
                name="get_weather",
                arguments='{"city":"广州"}',
            )
        ],
        output_text="",
        usage=SimpleNamespace(input_tokens=20, output_tokens=5, total_tokens=25),
        status="completed",
    )
    provider._client = SimpleNamespace(
        responses=SimpleNamespace(create=AsyncMock(return_value=fake_response)),
        chat=SimpleNamespace(completions=SimpleNamespace(create=AsyncMock())),
    )

    result = await provider.chat(messages=[{"role": "user", "content": "查广州天气"}])

    assert result.finish_reason == "tool_calls"
    assert result.tool_calls[0].id == "call_weather|fc_weather"
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == {"city": "广州"}
    assert result.usage == {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
    }
