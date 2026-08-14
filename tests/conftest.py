from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.models.appointment import Appointment
from app.models.channel import Channel
from app.models.conversation import Conversation
from app.models.knowledge_base import KnowledgeBase
from app.models.message import Message
from app.models.operator import Operator
from app.models.tenant import Tenant
from app.models.user import User
from app.repositories.appointment import AppointmentRepository
from app.repositories.channel import ChannelRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.repositories.message import MessageRepository
from app.repositories.operator import OperatorRepository
from app.repositories.tenant import TenantRepository
from app.repositories.user import UserRepository


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Wraps each test in an outer transaction (with a SAVEPOINT for any inner
    flush/commit) that's always rolled back afterward, so tests never leave
    data behind and never see another test's writes. Requires the Dockerized
    Postgres from docker-compose.yml to be reachable at DATABASE_URL.
    """
    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        trans = await connection.begin()
        session = AsyncSession(bind=connection, join_transaction_mode="create_savepoint")
        try:
            yield session
        finally:
            await session.close()
            await trans.rollback()
    await engine.dispose()


@pytest.fixture
def as_tenant() -> Callable[[UUID], AbstractContextManager[None]]:
    """`with as_tenant(tenant.id): ...` sets the current tenant for the block
    and always resets it afterward, even if the block raises.
    """

    @contextmanager
    def _as_tenant(tenant_id: UUID) -> Iterator[None]:
        token = set_current_tenant(tenant_id)
        try:
            yield
        finally:
            reset_current_tenant(token)

    return _as_tenant


@dataclass
class TenantSeed:
    channel: Channel
    user: User
    conversation: Conversation
    knowledge_base: KnowledgeBase
    operator: Operator
    appointment: Appointment
    message: Message


@dataclass
class Seed:
    tenant_a: Tenant
    tenant_b: Tenant
    a: TenantSeed
    b: TenantSeed


@pytest.fixture
async def seed(
    db_session: AsyncSession, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> Seed:
    """Creates two full tenants (A and B), each with one row in every
    tenant-scoped table plus a conversation + message, via the real
    repositories. Isolation tests use this as their starting data.
    """
    tenant_repo = TenantRepository(db_session)
    tenant_a = await tenant_repo.create(name="Clinic A", status="active")
    tenant_b = await tenant_repo.create(name="Clinic B", status="active")

    async def _build(tenant: Tenant) -> TenantSeed:
        with as_tenant(tenant.id):
            channel = await ChannelRepository(db_session).create(
                type="instagram", credentials="token", external_id=f"ig-{tenant.id}"
            )
            user = await UserRepository(db_session).create(
                channel_id=channel.id, external_id=f"ext-{tenant.id}"
            )
            conversation = await ConversationRepository(db_session).create(
                user_id=user.id, status="open"
            )
            knowledge_base = await KnowledgeBaseRepository(db_session).create(
                question="What are your hours?",
                answer="9 to 5.",
                embedding=[0.0] * 1536,
            )
            operator = await OperatorRepository(db_session).create(
                name="Dr. Smith", role="dentist", credentials="secret"
            )
            appointment = await AppointmentRepository(db_session).create(
                user_id=user.id,
                doctor="Dr. Smith",
                scheduled_at=datetime.now(UTC),
                status="scheduled",
            )
            message = await MessageRepository(db_session).create(
                conversation_id=conversation.id,
                sender="patient",
                content="Hello",
                channel="instagram",
            )
        return TenantSeed(
            channel=channel,
            user=user,
            conversation=conversation,
            knowledge_base=knowledge_base,
            operator=operator,
            appointment=appointment,
            message=message,
        )

    seed_a = await _build(tenant_a)
    seed_b = await _build(tenant_b)
    return Seed(tenant_a=tenant_a, tenant_b=tenant_b, a=seed_a, b=seed_b)
