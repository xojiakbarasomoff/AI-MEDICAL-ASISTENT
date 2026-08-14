import logging
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from arq.connections import RedisSettings
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.db import db_session
from app.core.redaction import preview
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.rag.embeddings import EmbeddingProvider
from app.rag.llm import LLMProvider
from app.services.answer import generate_answer

logger = logging.getLogger(__name__)


async def process_inbound_message(
    ctx: dict[str, Any],
    tenant_id: str,
    sender_igsid: str,
    message_text: str,
    *,
    session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]] = db_session,
    embedding_provider: EmbeddingProvider | None = None,
    llm_provider: LLMProvider | None = None,
) -> None:
    """ARQ job: generate a reply to one inbound Instagram message and (for
    now) log it.

    Runs in the worker process, not the webhook request — the request's DB
    session and tenant context don't survive past the 200 response, so this
    re-establishes both from the plain, serializable arguments the webhook
    enqueued (a UUID isn't one of those, hence tenant_id arrives as str and
    gets parsed back here).

    session_factory defaults to app.core.db.db_session (a genuinely fresh
    session against the real engine — "opens its own session", as in
    production this job never receives one from a caller). It's injectable
    for the same reason embedding_provider/llm_provider are: tests call this
    function directly, bypassing the queue, the same pattern used for
    generate_answer/ingest_faqs elsewhere in this codebase. This isn't just
    convenience — a *second*, independent db_session() in a test can't see
    that test's own uncommitted fixture data (Postgres won't show one
    connection another's uncommitted writes), so tests must be able to point
    the job at the same transactional session the test itself is using.
    """
    tenant_uuid = uuid.UUID(tenant_id)
    token = set_current_tenant(tenant_uuid)
    try:
        async with session_factory() as session:
            reply = await generate_answer(
                session,
                message_text,
                embedding_provider=embedding_provider,
                llm_provider=llm_provider,
            )
    finally:
        reset_current_tenant(token)

    # Full reply text stays out of INFO — it's patient-adjacent content that
    # shouldn't sit in logs that may ship to external monitoring (TZ section
    # 7, personal data). Length + truncated preview at INFO; full text only
    # at DEBUG, under a distinct event name so it's never ambiguous with the
    # redacted INFO line.
    logger.info(
        "webhook_reply_generated",
        extra={
            "tenant_id": tenant_id,
            "sender_igsid": sender_igsid,
            "reply_length": len(reply),
            "reply_preview": preview(reply),
        },
    )
    logger.debug(
        "webhook_reply_full_text",
        extra={"sender_igsid": sender_igsid, "reply": reply},
    )
    # TODO(IGB-?): call the Instagram Send API to deliver `reply` to
    # sender_igsid via this tenant's channel credentials, instead of just
    # logging it.


class WorkerSettings:
    functions = [process_inbound_message]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
