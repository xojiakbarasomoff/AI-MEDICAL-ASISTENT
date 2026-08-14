from functools import lru_cache

from pydantic import Field
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
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    webhook_verify_token: str = Field(alias="WEBHOOK_VERIFY_TOKEN")
    meta_app_secret: str = Field(alias="META_APP_SECRET")
    # TODO(IGB-?): move onto Tenant/per-tenant settings once the admin panel
    # (TZ 4.2) exists, so each clinic can tune its own debounce window
    # instead of every tenant sharing this one value — same pattern as
    # guardrail.EMERGENCY_RESPONSE / answer.NO_MATCH_RESPONSE.
    debounce_window_seconds: int = Field(default=25, alias="DEBOUNCE_WINDOW_SECONDS")


@lru_cache
def get_settings() -> Settings:
    return Settings()
