from fastapi import FastAPI

from app.api import webhook_router

app = FastAPI(title="Dental Clinic Instagram Assistant")
app.include_router(webhook_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
