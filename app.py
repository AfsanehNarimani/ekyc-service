import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from api.challenge_routes import router as challenge_router
from api.config_routes import router as config_router
from api.ekyc_routes import router as ekyc_router
from api.exception_handlers import request_validation_exception_handler
from api.video_verification_routes import router as video_verification_router
from api.readiness_routes import router as readiness_router
from config import settings

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(title=settings.app_name)

app.add_exception_handler(
    RequestValidationError,
    request_validation_exception_handler,
)

app.include_router(challenge_router)
app.include_router(config_router)
app.include_router(ekyc_router)
app.include_router(video_verification_router)
app.include_router(readiness_router)


@app.get("/health")
def health_check():
    return {"ok": True, "service": settings.app_name}
