from fastapi import FastAPI

app = FastAPI(title="Dental Clinic Instagram Assistant")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
