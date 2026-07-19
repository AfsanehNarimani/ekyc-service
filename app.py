from fastapi import FastAPI

from api.challenge_routes import router as challenge_router
from api.config_routes import router as config_router
from api.ekyc_routes import router as ekyc_router
from api.video_verification_routes import router as video_verification_router
from config import settings

app = FastAPI(
    title=settings.app_name,
)

app.include_router(challenge_router)
app.include_router(config_router)
app.include_router(ekyc_router)
app.include_router(video_verification_router)


@app.get("/health")
def health_check():
    return {
        "ok": True,
        "service": settings.app_name,
    }
