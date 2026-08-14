from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.rag.embeddings import EMBEDDING_DIMENSIONS, OpenAIEmbeddingProvider

TEST_SETTINGS = Settings(
    database_url="postgresql+asyncpg://test:test@localhost/test",
    redis_url="redis://localhost:6379/0",
    openai_api_key="sk-test",
    webhook_verify_token="test-verify-token",
    meta_app_secret="test-app-secret",
)


class _FakeEmbeddingsResource:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.create = AsyncMock(
            return_value=SimpleNamespace(
                data=[SimpleNamespace(embedding=vector) for vector in vectors]
            )
        )


class _FakeAsyncOpenAI:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = _FakeEmbeddingsResource(vectors)


# No test here ever talks to the real OpenAI API: AsyncOpenAI is monkeypatched
# at the point embeddings.py imports it, so OpenAIEmbeddingProvider.__init__
# picks up the fake client instead of a real network client.


async def test_embed_returns_vectors_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    vectors = [[0.1] * EMBEDDING_DIMENSIONS, [0.2] * EMBEDDING_DIMENSIONS]
    fake_client = _FakeAsyncOpenAI(vectors)
    monkeypatch.setattr("app.rag.embeddings.AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAIEmbeddingProvider(settings=TEST_SETTINGS)
    result = await provider.embed(["hello", "world"])

    assert result == vectors
    fake_client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small",
        input=["hello", "world"],
        dimensions=EMBEDDING_DIMENSIONS,
    )


async def test_embed_empty_list_skips_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeAsyncOpenAI([])
    monkeypatch.setattr("app.rag.embeddings.AsyncOpenAI", lambda **kwargs: fake_client)

    provider = OpenAIEmbeddingProvider(settings=TEST_SETTINGS)
    result = await provider.embed([])

    assert result == []
    fake_client.embeddings.create.assert_not_awaited()
