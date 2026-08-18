import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Appointment(Base):
    __tablename__ = "appointments"
    __table_args__ = (
        # A slot counts as "taken" only while status='scheduled' — this is
        # partial, not a plain unique constraint, so cancelling (or a future
        # no-show/completed transition) drops the row out of the index and
        # frees the slot for rebooking. This index is the actual
        # double-booking guard: check_availability()/find_next_free_slot()
        # in app.services.appointment are UX only (they can go stale between
        # the check and the insert); this constraint is what makes a race
        # between two simultaneous bookings resolve to exactly one winner.
        Index(
            "uq_appointments_tenant_id_scheduled_at",
            "tenant_id",
            "scheduled_at",
            unique=True,
            postgresql_where=text("status = 'scheduled'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True, index=True
    )
    doctor: Mapped[str] = mapped_column(String(255), nullable=False)
    # The name to book under — may differ from user.name (the IG account
    # holder may be booking for a spouse/child). Falls back to user.name in
    # the service layer when omitted.
    patient_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    # "bot" | "operator" — which side created this booking.
    source: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=text("'operator'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), onupdate=func.now(), nullable=False
    )
