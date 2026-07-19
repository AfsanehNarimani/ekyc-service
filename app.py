from fastapi import FastAPI

from config import settings

app = FastAPI(
    title=settings.app_name,
)


@app.get("/health")
def health_check():
    return {
        "ok": True,
        "service": settings.app_name,
    }
