from abc import ABC, abstractmethod
from functools import lru_cache

from openai import AsyncOpenAI

from app.core.config import Settings, get_settings

EMBEDDING_DIMENSIONS = 1536


class EmbeddingProvider(ABC):
    """Abstraction over "turn text into vectors", mirroring the TZ's
    LLMProvider idea so the concrete backend (OpenAI today, something else
    later) can be swapped without touching callers. Callers should depend on
    this interface, not on OpenAIEmbeddingProvider directly, so tests can
    inject a fake instead of hitting the network.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each text, returning one vector per input in the same order."""


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        settings: Settings | None = None,
        model: str = "text-embedding-3-small",
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=(settings or get_settings()).openai_api_key)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # A single batched call, not one call per text: OpenAI's embeddings
        # endpoint accepts a list under `input` and returns vectors in the
        # same order, so batching here saves N-1 round trips per ingest.
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        return [item.embedding for item in response.data]


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return OpenAIEmbeddingProvider()
