import hashlib
import hmac
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.db import get_db_session
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.services.tenant_resolution import resolve_tenant_for_ig_account

logger = logging.getLogger(__name__)

router = APIRouter()


class WebhookSender(BaseModel):
    id: str


class WebhookRecipient(BaseModel):
    id: str


class WebhookMessage(BaseModel):
    text: str | None = None
    is_echo: bool = False


class MessagingEvent(BaseModel):
    sender: WebhookSender
    recipient: WebhookRecipient
    message: WebhookMessage | None = None


class WebhookEntry(BaseModel):
    id: str
    messaging: list[MessagingEvent] = []


class WebhookPayload(BaseModel):
    object: str
    entry: list[WebhookEntry] = []


def _verify_signature(raw_body: bytes, signature_header: str | None, app_secret: str) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    provided = signature_header.removeprefix("sha256=")
    return hmac.compare_digest(expected, provided)


def _is_echo(event: MessagingEvent, page_id: str) -> bool:
    if event.message is not None and event.message.is_echo:
        return True
    return event.sender.id == page_id


async def _handle_payload(session: AsyncSession, payload: WebhookPayload) -> None:
    for entry in payload.entry:
        # entry.id is the IG account (page) id that received the message —
        # one tenant's channel per entry, so resolution happens once per
        # entry rather than once per messaging event.
        tenant_id = await resolve_tenant_for_ig_account(session, entry.id)
        if tenant_id is None:
            logger.warning("webhook_unknown_ig_account", extra={"ig_account_id": entry.id})
            continue

        token = set_current_tenant(tenant_id)
        try:
            for event in entry.messaging:
                if event.message is None:
                    # Not a message event (e.g. read receipt, postback) — nothing to do yet.
                    continue

                if _is_echo(event, entry.id):
                    logger.info(
                        "webhook_echo_skipped",
                        extra={"sender_id": event.sender.id, "recipient_id": event.recipient.id},
                    )
                    continue

                logger.info(
                    "webhook_message_received",
                    extra={
                        "tenant_id": str(tenant_id),
                        "sender_igsid": event.sender.id,
                        "recipient_id": event.recipient.id,
                        "message_text": event.message.text,
                    },
                )

                # TODO(IGB-?): enqueue the message onto Redis/ARQ for async processing
                # TODO(IGB-?): call the RAG/LLM pipeline to generate and send a reply
        finally:
            reset_current_tenant(token)


@router.get("/webhook")
async def verify_webhook(
    hub_mode: str | None = Query(default=None, alias="hub.mode"),
    hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
    settings: Settings = Depends(get_settings),
) -> Response:
    if hub_mode == "subscribe" and hub_verify_token == settings.webhook_verify_token:
        return Response(content=hub_challenge or "", media_type="text/plain")
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Webhook verification failed")


@router.post("/webhook")
async def receive_webhook(
    request: Request,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_db_session),
) -> Response:
    raw_body = await request.body()
    signature_header = request.headers.get("x-hub-signature-256")

    if not _verify_signature(raw_body, signature_header, settings.meta_app_secret):
        logger.warning("webhook_signature_invalid")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid signature")

    try:
        payload = WebhookPayload.model_validate_json(raw_body)
    except ValueError:
        logger.warning("webhook_payload_invalid")
        return Response(status_code=status.HTTP_200_OK)

    await _handle_payload(session, payload)

    return Response(status_code=status.HTTP_200_OK)
