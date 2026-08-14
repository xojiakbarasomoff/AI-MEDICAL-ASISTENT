from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.rag.llm import OpenAILLMProvider

TEST_SETTINGS = Settings(
    database_url="postgresql+asyncpg://test:test@localhost/test",
    redis_url="redis://localhost:6379/0",
    openai_api_key="sk-test",
    webhook_verify_token="test-verify-token",
    meta_app_secret="test-app-secret",
)


class _FakeCompletionsResource:
    def __init__(self, content: str | None) -> None:
        self.create = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )
        )


class _FakeChat:
    def __init__(self, content: str | None) -> None:
        self.completions = _FakeCompletionsResource(content)


class _FakeAsyncOpenAI:
    def __init__(self, content: str | None) -> None:
        self.chat = _FakeChat(content)


# No test here ever talks to the real OpenAI API: AsyncOpenAI is monkeypatched
# at the point llm.py imports it, so OpenAILLMProvider.__init__ picks up the
# fake client instead of a real network client.


async def test_generate_prepends_system_prompt_and_returns_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeAsyncOpenAI("Sure, we're open 9 to 5!")
    monkeypatch.setattr("app.rag.llm.AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAILLMProvider(settings=TEST_SETTINGS)
    result = await provider.generate(
        "You are a helpful assistant.", [{"role": "user", "content": "What are your hours?"}]
    )

    assert result == "Sure, we're open 9 to 5!"
    fake_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are your hours?"},
        ],
    )


async def test_generate_raises_when_content_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeAsyncOpenAI(None)
    monkeypatch.setattr("app.rag.llm.AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAILLMProvider(settings=TEST_SETTINGS)
    with pytest.raises(ValueError, match="no text content"):
        await provider.generate("system", [{"role": "user", "content": "hi"}])
