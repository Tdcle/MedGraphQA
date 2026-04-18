import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, chat, health, memory
from app.core.config import settings
from app.core.container import build_container
from app.core.logging import configure_logging
from app.middleware.request_logging import RequestLoggingMiddleware


configure_logging(settings)
logger = logging.getLogger("medgraphqa.app")

app = FastAPI(title=settings.api_title, version=settings.api_version, debug=settings.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    RequestLoggingMiddleware,
    access_log_enabled=settings.access_log_enabled,
)

app.state.container = build_container(settings)
logger.info(
    "backend started provider=%s model=%s python=%s",
    settings.llm_provider,
    settings.ollama_model if settings.llm_provider == "ollama" else settings.dashscope_model,
    sys.executable,
)

app.include_router(health.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(memory.router, prefix="/api")


@app.on_event("shutdown")
def shutdown() -> None:
    app.state.container.entity_repository.close()
    app.state.container.auth_repository.close()
    app.state.container.memory_repository.close()


@app.get("/")
def root():
    return {"message": "MedGraphQA FastAPI backend is running"}
