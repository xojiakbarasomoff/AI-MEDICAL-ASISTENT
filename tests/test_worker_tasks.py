import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, AbstractContextManager, asynccontextmanager
from uuid import UUID

import pytest
from arq.connections import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession

from app.rag.embeddings import EMBEDDING_DIMENSIONS, EmbeddingProvider
from app.rag.llm import ChatMessage, LLMProvider
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.workers.tasks import fire_debounce_window, process_inbound_message
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
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        self.calls.append((system_prompt, messages))
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


# --- fire_debounce_window: window fires once after quiet period ---


async def test_fire_debounce_window_processes_joined_batch_when_generation_current(
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
    redis_pool: ArqRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reply_text = "We're open 9 to 5, Monday to Saturday."
    embedding_provider = FakeEmbeddingProvider(QUERY_VECTOR)
    llm_provider = FakeLLMProvider(reply=reply_text)
    sender_igsid = "sender-1"

    with as_tenant(seed.tenant_a.id):
        await KnowledgeBaseRepository(db_session).create(
            question="What are your hours?", answer="9 to 5, Mon-Sat.", embedding=QUERY_VECTOR
        )

    messages_key = f"debounce:{seed.tenant_a.id}:{sender_igsid}:messages"
    generation_key = f"debounce:{seed.tenant_a.id}:{sender_igsid}:generation"
    await redis_pool.rpush(messages_key, "What time do you open?", "Also, are you open Sundays?")
    await redis_pool.set(generation_key, "3")

    with caplog.at_level(logging.INFO, logger="app.workers.tasks"):
        await fire_debounce_window(
            {"redis": redis_pool},
            str(seed.tenant_a.id),
            sender_igsid,
            3,
            session_factory=_session_factory(db_session),
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
        )

    # The batch was claimed and cleared, generate_answer ran once on the
    # joined text (order preserved), and the reply got logged (redacted).
    assert await redis_pool.exists(messages_key) == 0
    assert await redis_pool.exists(generation_key) == 0
    assert len(llm_provider.calls) == 1
    _system_prompt, messages = llm_provider.calls[0]
    assert messages == [
        {
            "role": "user",
            "content": "What time do you open?\nAlso, are you open Sundays?",
        }
    ]
    [info_record] = [r for r in caplog.records if r.message == "webhook_reply_generated"]
    assert info_record.tenant_id == str(seed.tenant_a.id)  # type: ignore[attr-defined]
    assert info_record.sender_igsid == sender_igsid  # type: ignore[attr-defined]
    assert info_record.reply_preview == reply_text  # type: ignore[attr-defined]


# --- fire_debounce_window: stale generation jobs no-op ---


async def test_fire_debounce_window_stale_generation_does_not_process_or_log(
    seed: Seed,
    redis_pool: ArqRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm_provider = FakeLLMProvider(reply="should never be used")
    sender_igsid = "sender-1"

    messages_key = f"debounce:{seed.tenant_a.id}:{sender_igsid}:messages"
    generation_key = f"debounce:{seed.tenant_a.id}:{sender_igsid}:generation"
    await redis_pool.rpush(messages_key, "first message")
    await redis_pool.set(generation_key, "2")  # a second message already reset the window

    with caplog.at_level(logging.INFO, logger="app.workers.tasks"):
        await fire_debounce_window(
            {"redis": redis_pool},
            str(seed.tenant_a.id),
            sender_igsid,
            1,  # stale — the live generation is 2
            llm_provider=llm_provider,
        )

    assert llm_provider.calls == []
    assert caplog.records == []
    # Untouched — the still-current generation's buffer must survive so it
    # can fire correctly later.
    messages = await redis_pool.lrange(messages_key, 0, -1)
    assert [m.decode() for m in messages] == ["first message"]
    assert await redis_pool.get(generation_key) == b"2"
