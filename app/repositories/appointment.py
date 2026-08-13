from app.models.appointment import Appointment
from app.repositories.base import TenantScopedRepository


class AppointmentRepository(TenantScopedRepository[Appointment]):
    model = Appointment
