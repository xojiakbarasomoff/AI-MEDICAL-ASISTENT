from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Literal, TypedDict, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from app.core.config import Settings, get_settings


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class LLMProvider(ABC):
    """Abstraction over "turn a system prompt + conversation into a reply",
    mirroring EmbeddingProvider so the backend/model can change without
    touching callers, and so tests can inject a fake instead of hitting the
    network.
    """

    @abstractmethod
    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        """Generate a reply given a system prompt and the conversation so far."""


class OpenAILLMProvider(LLMProvider):
    def __init__(self, settings: Settings | None = None, model: str = "gpt-4o-mini") -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=(settings or get_settings()).openai_api_key)

    async def generate(self, system_prompt: str, messages: list[ChatMessage]) -> str:
        # ChatMessage is deliberately narrower than the SDK's message union
        # (only user/assistant; system is handled separately by this
        # abstraction), so it doesn't structurally unify with
        # ChatCompletionMessageParam under mypy strict. Both sides are
        # simple {role, content} dicts at runtime, so the cast is safe.
        payload = cast(
            "list[ChatCompletionMessageParam]",
            [{"role": "system", "content": system_prompt}, *messages],
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=payload,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAI chat completion returned no text content")
        return content


@lru_cache
def get_llm_provider() -> LLMProvider:
    return OpenAILLMProvider()
