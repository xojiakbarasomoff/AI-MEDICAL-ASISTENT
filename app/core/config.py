from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str = Field(alias="DATABASE_URL")
    redis_url: str = Field(alias="REDIS_URL")
    webhook_verify_token: str = Field(alias="WEBHOOK_VERIFY_TOKEN")
    meta_app_secret: str = Field(alias="META_APP_SECRET")
    # TODO(IGB-?): move onto Tenant/per-tenant settings once the admin panel
    # (TZ 4.2) exists, so each clinic can tune its own debounce window
    # instead of every tenant sharing this one value — same pattern as
    # guardrail.EMERGENCY_RESPONSE / answer.NO_MATCH_RESPONSE.
    debounce_window_seconds: int = Field(default=25, alias="DEBOUNCE_WINDOW_SECONDS")

    # Single switch governing both the LLM and embedding backend (see
    # app.rag.llm._select_llm_provider / app.rag.embeddings._select_embedding_provider).
    # Defaults to gemini: the CEO's direction is Gemini, and we may have no
    # OpenAI key at all — defaulting to openai would fail with a 401 the
    # moment anyone ran this without explicitly setting the provider. openai
    # stays fully implemented and selectable in case we switch back.
    model_provider: Literal["openai", "gemini"] = Field(default="gemini", alias="MODEL_PROVIDER")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")

    @model_validator(mode="after")
    def _require_active_provider_key(self) -> Self:
        if self.model_provider == "openai" and self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai")
        if self.model_provider == "gemini" and self.gemini_api_key is None:
            raise ValueError("GEMINI_API_KEY is required when MODEL_PROVIDER=gemini")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
