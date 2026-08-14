import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager, asynccontextmanager
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embeddings import EMBEDDING_DIMENSIONS, EmbeddingProvider
from app.rag.llm import ChatMessage, LLMProvider
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.workers.tasks import process_inbound_message
from tests.conftest import Seed

QUERY_VECTOR = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


class FakeLLMProvider(LLMProvider):
    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        return self._reply


def _session_factory(
    session: AsyncSession,
) -> Callable[[], AbstractAsyncContextManager[AsyncSession]]:
    """Wraps this test's own transactional db_session as the factory
    process_inbound_message calls, instead of it opening a genuinely new
    connection — which couldn't see this test's uncommitted fixture data
    anyway, and would be bound to a different event loop than pytest-asyncio
    hands this test.
    """

    @asynccontextmanager
    async def _factory() -> AsyncIterator[AsyncSession]:
        yield session

    return _factory


async def test_process_inbound_message_runs_under_correct_tenant_and_logs_redacted_reply(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    reply_text = "We're open 9 to 5, Monday to Saturday."
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider(reply=reply_text)

    # Seeded (and left) under tenant A's context, then the job is called
    # with NO ambient tenant set — proving it re-establishes tenant context
    # itself from the plain tenant_id argument, rather than relying on one
    # already bound in the caller's context.
    with as_tenant(seed.tenant_a.id):
        await KnowledgeBaseRepository(db_session).create(
            question="What are your hours?", answer="9 to 5, Mon-Sat.", embedding=QUERY_VECTOR
        )

    with caplog.at_level(logging.DEBUG, logger="app.workers.tasks"):
        await process_inbound_message(
            {},
            str(seed.tenant_a.id),
            "sender-1",
            "What time do you open?",
            session_factory=_session_factory(db_session),
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

    [info_record] = [r for r in caplog.records if r.levelname == "INFO"]
    assert info_record.message == "webhook_reply_generated"
    assert info_record.tenant_id == str(seed.tenant_a.id)  # type: ignore[attr-defined]
    assert info_record.sender_igsid == "sender-1"  # type: ignore[attr-defined]
    assert info_record.reply_length == len(reply_text)  # type: ignore[attr-defined]
    assert info_record.reply_preview == reply_text  # type: ignore[attr-defined]
    # Full reply text must not appear anywhere on the INFO record.
    assert "reply" not in info_record.__dict__

    [debug_record] = [r for r in caplog.records if r.levelname == "DEBUG"]
    assert debug_record.message == "webhook_reply_full_text"
    assert debug_record.reply == reply_text  # type: ignore[attr-defined]


async def test_process_inbound_message_reply_preview_is_truncated_for_long_reply(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    reply_text = "A" * 100
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider(reply=reply_text)

    with as_tenant(seed.tenant_a.id):
        await KnowledgeBaseRepository(db_session).create(
            question="What are your hours?", answer="9 to 5, Mon-Sat.", embedding=QUERY_VECTOR
        )

    with caplog.at_level(logging.INFO, logger="app.workers.tasks"):
        await process_inbound_message(
            {},
            str(seed.tenant_a.id),
            "sender-1",
            "What time do you open?",
            session_factory=_session_factory(db_session),
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

    [info_record] = [r for r in caplog.records if r.levelname == "INFO"]
    assert info_record.reply_length == 100  # type: ignore[attr-defined]
    assert info_record.reply_preview == "A" * 40 + "…"  # type: ignore[attr-defined]
