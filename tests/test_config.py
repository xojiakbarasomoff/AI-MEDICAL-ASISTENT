import pytest
from pydantic import ValidationError

from app.core.config import Settings

_BASE_KWARGS = {
    "database_url": "postgresql+asyncpg://test:test@localhost/test",
    "redis_url": "redis://localhost:6379/0",
    "webhook_verify_token": "test-verify-token",
    "meta_app_secret": "test-app-secret",
}


def test_defaults_to_gemini_provider() -> None:
    settings = Settings(**_BASE_KWARGS, gemini_api_key="test-gemini-key")
    assert settings.model_provider == "gemini"


def test_gemini_provider_without_gemini_key_raises() -> None:
    # gemini_api_key must be forced to None explicitly, not just omitted:
    # Settings falls back to the real GEMINI_API_KEY env var for any field
    # not given a constructor value, which would silently defeat this test
    # in an environment (like CI, or this test run) that has one set.
    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        Settings(**_BASE_KWARGS, model_provider="gemini", gemini_api_key=None)


def test_openai_provider_without_openai_key_raises() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(
            **_BASE_KWARGS,
            model_provider="openai",
            openai_api_key=None,
            gemini_api_key="test-gemini-key",
        )


def test_openai_provider_with_openai_key_is_valid() -> None:
    settings = Settings(**_BASE_KWARGS, model_provider="openai", openai_api_key="sk-test")
    assert settings.model_provider == "openai"


def test_gemini_provider_with_gemini_key_is_valid() -> None:
    settings = Settings(**_BASE_KWARGS, model_provider="gemini", gemini_api_key="test-gemini-key")
    assert settings.model_provider == "gemini"
