from app.models.knowledge_base import KnowledgeBase
from app.repositories.base import TenantScopedRepository


class KnowledgeBaseRepository(TenantScopedRepository[KnowledgeBase]):
    model = KnowledgeBase
