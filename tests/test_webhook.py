import hashlib
import hmac
import json
import logging
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app

TEST_SETTINGS = Settings(
    database_url="postgresql+asyncpg://test:test@localhost/test",
    redis_url="redis://localhost:6379/0",
    openai_api_key="sk-test",
    webhook_verify_token="test-verify-token",
    meta_app_secret="test-app-secret",
)


@pytest.fixture(autouse=True)
def _override_settings() -> Iterator[None]:
    app.dependency_overrides[get_settings] = lambda: TEST_SETTINGS
    yield
    app.dependency_overrides.pop(get_settings, None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _sign(body: bytes, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


# --- GET /webhook verification handshake ---


def test_verify_webhook_with_valid_token_returns_challenge(client: TestClient) -> None:
    response = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "test-verify-token",
            "hub.challenge": "12345",
        },
    )
    assert response.status_code == 200
    assert response.text == "12345"


def test_verify_webhook_with_invalid_token_returns_403(client: TestClient) -> None:
    response = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong-token",
            "hub.challenge": "12345",
        },
    )
    assert response.status_code == 403


def test_verify_webhook_with_wrong_mode_returns_403(client: TestClient) -> None:
    response = client.get(
        "/webhook",
        params={
            "hub.mode": "unsubscribe",
            "hub.verify_token": "test-verify-token",
            "hub.challenge": "12345",
        },
    )
    assert response.status_code == 403


# --- POST /webhook signature validation ---


def test_receive_webhook_with_valid_signature_returns_200(client: TestClient) -> None:
    body = json.dumps({"object": "instagram", "entry": []}).encode("utf-8")
    signature = _sign(body, "test-app-secret")

    response = client.post("/webhook", content=body, headers={"X-Hub-Signature-256": signature})
    assert response.status_code == 200


def test_receive_webhook_with_invalid_signature_returns_403(client: TestClient) -> None:
    body = json.dumps({"object": "instagram", "entry": []}).encode("utf-8")

    response = client.post(
        "/webhook", content=body, headers={"X-Hub-Signature-256": "sha256=" + "0" * 64}
    )
    assert response.status_code == 403


def test_receive_webhook_with_missing_signature_returns_403(client: TestClient) -> None:
    body = json.dumps({"object": "instagram", "entry": []}).encode("utf-8")

    response = client.post("/webhook", content=body)
    assert response.status_code == 403


# --- Echo filtering ---


def _messaging_payload(
    sender_id: str, recipient_id: str, text: str, is_echo: bool = False
) -> bytes:
    payload = {
        "object": "instagram",
        "entry": [
            {
                "id": "page-1",
                "messaging": [
                    {
                        "sender": {"id": sender_id},
                        "recipient": {"id": recipient_id},
                        "message": {"text": text, "is_echo": is_echo},
                    }
                ],
            }
        ],
    }
    return json.dumps(payload).encode("utf-8")


def test_receive_webhook_skips_echo_event(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    body = _messaging_payload("page-1", "user-1", "We'll be right with you", is_echo=True)
    signature = _sign(body, "test-app-secret")

    with caplog.at_level(logging.INFO, logger="app.api.webhook"):
        response = client.post("/webhook", content=body, headers={"X-Hub-Signature-256": signature})

    assert response.status_code == 200
    assert "webhook_echo_skipped" in caplog.text
    assert "webhook_message_received" not in caplog.text


def test_receive_webhook_skips_event_from_own_account_without_echo_flag(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    body = _messaging_payload("page-1", "user-1", "Auto message", is_echo=False)
    signature = _sign(body, "test-app-secret")

    with caplog.at_level(logging.INFO, logger="app.api.webhook"):
        response = client.post("/webhook", content=body, headers={"X-Hub-Signature-256": signature})

    assert response.status_code == 200
    assert "webhook_echo_skipped" in caplog.text


def test_receive_webhook_processes_genuine_inbound_message(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    body = _messaging_payload("user-1", "page-1", "Hi, do you have an opening tomorrow?")
    signature = _sign(body, "test-app-secret")

    with caplog.at_level(logging.INFO, logger="app.api.webhook"):
        response = client.post("/webhook", content=body, headers={"X-Hub-Signature-256": signature})

    assert response.status_code == 200
    assert "webhook_message_received" in caplog.text
    assert "webhook_echo_skipped" not in caplog.text
