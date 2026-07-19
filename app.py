from fastapi import FastAPI

from api.challenge_routes import router as challenge_router
from api.ekyc_routes import router as ekyc_router
from config import settings

app = FastAPI(
    title=settings.app_name,
)

app.include_router(challenge_router)
app.include_router(ekyc_router)


@app.get("/health")
def health_check():
    return {
        "ok": True,
        "service": settings.app_name,
    }
