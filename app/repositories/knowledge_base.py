from typing import Any

from app.core.tenant_context import get_current_tenant
from app.models.knowledge_base import KnowledgeBase
from app.repositories.base import CrossTenantAccessError, TenantScopedRepository


class KnowledgeBaseRepository(TenantScopedRepository[KnowledgeBase]):
    model = KnowledgeBase

    async def get_by_question(self, question: str) -> KnowledgeBase | None:
        """Exact-match lookup scoped to the current tenant, used to decide
        whether a FAQ import is a new row or an update to an existing one.
        """
        return await self._get(question=question, tenant_id=get_current_tenant())

    async def update(self, obj: KnowledgeBase, **values: Any) -> KnowledgeBase:
        """Mutates and flushes an already-loaded row.

        Callers are expected to have fetched `obj` through a tenant-scoped
        read (e.g. get_by_question()), but this doesn't trust that: it
        re-checks obj.tenant_id against the current tenant itself, the same
        way create() re-checks a caller-supplied tenant_id via
        _resolve_tenant_id(), so a future caller that skips the scoped fetch
        fails loudly instead of silently writing across tenants.
        """
        current_tenant_id = get_current_tenant()
        if obj.tenant_id != current_tenant_id:
            raise CrossTenantAccessError(
                f"obj.tenant_id={obj.tenant_id} does not match the current tenant "
                f"({current_tenant_id})"
            )
        for field, value in values.items():
            setattr(obj, field, value)
        await self.session.flush()
        return obj
