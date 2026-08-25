from __future__ import annotations

import inspect
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from dwell import __version__
from dwell.config import DwellConfig
from dwell.domain import JobRecord, JobStatus, Modality, ModelView, VideoGenerationRequest
from dwell.errors import DwellError
from dwell.jobs import JobManager, JobStore
from dwell.logging_config import configure_logging

logger = logging.getLogger(__name__)


class VideoJobAccepted(BaseModel):
    id: str
    status: JobStatus


def _error(code: str, message: str, details: Any = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details}}


def _validation_details(exc: RequestValidationError) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for raw_error in exc.errors():
        error = dict(raw_error)
        error.pop("url", None)
        context = error.get("ctx")
        if isinstance(context, dict):
            error["ctx"] = {key: str(value) for key, value in context.items()}
        details.append(jsonable_encoder(error))
    return details


def _build_model_manager(
    config: DwellConfig,
    *,
    model_registry: Any = None,
    runtime_registry: Any = None,
) -> Any:
    # Imports stay local so repository-only consumers can use the job store
    # without importing optional runtime integration code.
    from dwell.model_manager import ModelManager
    from dwell.registry import ModelRegistry
    from dwell.runtimes.registry import RuntimeRegistry

    registry = model_registry or ModelRegistry(config)
    runtimes = runtime_registry or RuntimeRegistry(config, registry)
    return ModelManager(config, registry=registry, runtimes=runtimes)


def create_app(
    config: DwellConfig | None = None,
    *,
    model_manager: Any = None,
    runtime_registry: Any = None,
    model_registry: Any = None,
    job_store: JobStore | None = None,
    job_manager: JobManager | None = None,
) -> FastAPI:
    """Build the local API with replaceable model/runtime/job dependencies."""

    resolved_config = config or DwellConfig.from_env()
    manager = model_manager or _build_model_manager(
        resolved_config,
        model_registry=model_registry,
        runtime_registry=runtime_registry,
    )
    store = job_store or (job_manager.store if job_manager is not None else None)
    store = store or JobStore(resolved_config.jobs_db)
    jobs = job_manager or JobManager(store, manager)
    started_at: datetime | None = None

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        nonlocal started_at
        resolved_config.ensure_layout()
        configure_logging(resolved_config)
        started_at = datetime.now(UTC)
        logger.info("Dwell API starting on %s", resolved_config.api_url)
        await jobs.start()
        try:
            yield
        finally:
            logger.info("Dwell API shutting down")
            try:
                await jobs.stop()
            finally:
                shutdown = getattr(manager, "shutdown", None)
                if callable(shutdown):
                    result = shutdown()
                    if inspect.isawaitable(result):
                        await result

    application = FastAPI(title="Dwell", version=__version__, lifespan=lifespan)
    application.state.config = resolved_config
    application.state.model_manager = manager
    application.state.job_store = store
    application.state.job_manager = jobs

    from dwell.api.openai import create_openai_router

    application.include_router(create_openai_router(manager))

    @application.exception_handler(DwellError)
    async def dwell_error_handler(_request: Request, exc: DwellError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(exc.as_dict()))

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=_error(
                "invalid_request",
                "The request is invalid.",
                _validation_details(exc),
            ),
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        _request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        code = "not_found" if exc.status_code == status.HTTP_404_NOT_FOUND else "invalid_request"
        return JSONResponse(
            status_code=exc.status_code,
            content=_error(code, str(exc.detail)),
            headers=exc.headers,
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error("internal_error", "An unexpected internal error occurred."),
        )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/v1/status")
    async def service_status() -> dict[str, Any]:
        model_views = manager.list_models()
        active = store.active()
        result: dict[str, Any] = {
            "server": "running",
            "version": __version__,
            "loaded_models": [model.id for model in model_views if model.loaded],
            "active_job": active.id if active is not None else None,
            "queued_jobs": store.count(JobStatus.QUEUED),
        }
        if started_at is not None:
            result["started_at"] = started_at.isoformat()
        return result

    @application.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "data": [model.model_dump(mode="json") for model in manager.list_models()],
        }

    @application.post("/v1/models/{model_id}/load", response_model=ModelView)
    async def load_model(model_id: str) -> ModelView:
        return await manager.load(model_id)

    @application.delete("/v1/models/{model_id}/load", response_model=ModelView)
    async def unload_model(model_id: str) -> ModelView:
        return await manager.unload(model_id)

    @application.delete("/v1/models", response_model=list[ModelView])
    async def unload_all_models() -> list[ModelView]:
        return await manager.unload_all()

    @application.post(
        "/v1/videos",
        response_model=VideoJobAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_video(request: VideoGenerationRequest) -> VideoJobAccepted:
        # This is deliberately before persistence. ensure_installed is local-only,
        # and inference never doubles as installation.
        model = manager.ensure_installed(request.model)
        if model.modality != Modality.VIDEO:
            raise DwellError(
                "invalid_request",
                f"Model '{request.model}' does not support video generation.",
                details={"modality": model.modality.value},
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            )
        if not model.available:
            raise DwellError(
                "runtime_not_available",
                f"Runtime '{model.runtime}' is not available locally.",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        job = jobs.submit_video(request)
        return VideoJobAccepted(id=job.id, status=job.status)

    @application.get("/v1/jobs/{job_id}", response_model=JobRecord)
    async def get_job(job_id: str) -> JobRecord:
        return store.get(job_id)

    @application.delete("/v1/jobs/{job_id}", response_model=JobRecord)
    async def cancel_job(job_id: str) -> JobRecord:
        return await jobs.cancel(job_id)

    return application
