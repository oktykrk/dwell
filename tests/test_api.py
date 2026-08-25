from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from dwell.api import create_app
from dwell.config import DwellConfig
from dwell.domain import (
    GenerationResult,
    Modality,
    ModelState,
    ModelView,
)
from dwell.errors import model_not_installed


def model_view(*, installed: bool, available: bool | None = None) -> ModelView:
    if available is None:
        available = installed
    return ModelView(
        id="local-video",
        family="test",
        version="1",
        modality=Modality.VIDEO,
        runtime="fake",
        quantization=None,
        installed=installed,
        available=available,
        loaded=False,
        state=ModelState.INSTALLED if installed else ModelState.NOT_INSTALLED,
    )


class FakeModelManager:
    def __init__(self, *, installed: bool, available: bool | None = None) -> None:
        self.view = model_view(installed=installed, available=available)
        self.generate_calls = 0

    def list_models(self) -> list[ModelView]:
        return [self.view]

    def ensure_installed(self, model_id: str) -> ModelView:
        assert model_id == self.view.id
        if not self.view.installed:
            raise model_not_installed(model_id)
        return self.view

    async def generate_video(self, _request, job_id: str) -> GenerationResult:
        self.generate_calls += 1
        return GenerationResult(
            output_path=Path(f"/tmp/{job_id}.mp4"),
            duration_seconds=0.1,
        )


def test_health_status_and_models_are_truthful_and_offline(tmp_path: Path) -> None:
    manager = FakeModelManager(installed=False)
    app = create_app(DwellConfig(home=tmp_path), model_manager=manager)

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        service_status = client.get("/v1/status")
        assert service_status.status_code == 200
        assert service_status.json()["server"] == "running"
        assert service_status.json()["loaded_models"] == []
        assert service_status.json()["active_job"] is None
        assert service_status.json()["queued_jobs"] == 0

        models = client.get("/v1/models")
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "local-video"
        assert models.json()["data"][0]["installed"] is False

    log = (tmp_path / "logs" / "dwell.log").read_text(encoding="utf-8")
    assert "Dwell API starting" in log
    assert "Dwell API shutting down" in log


def test_missing_model_fails_before_queue_and_never_runs(tmp_path: Path) -> None:
    manager = FakeModelManager(installed=False)
    app = create_app(DwellConfig(home=tmp_path), model_manager=manager)

    with TestClient(app) as client:
        response = client.post(
            "/v1/videos",
            json={"model": "local-video", "prompt": "No implicit download"},
        )
        assert response.status_code == 409
        assert response.json() == {
            "error": {
                "code": "model_not_installed",
                "message": "Model 'local-video' is not installed locally.",
                "details": None,
            }
        }
        assert client.get("/v1/status").json()["queued_jobs"] == 0
        assert app.state.job_store.list() == []
        assert manager.generate_calls == 0


def test_unavailable_runtime_fails_before_queue_and_never_runs(tmp_path: Path) -> None:
    manager = FakeModelManager(installed=True, available=False)
    app = create_app(DwellConfig(home=tmp_path), model_manager=manager)

    with TestClient(app) as client:
        response = client.post(
            "/v1/videos",
            json={"model": "local-video", "prompt": "No unavailable runtime job"},
        )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "runtime_not_available"
        assert app.state.job_store.list() == []
        assert manager.generate_calls == 0


def test_pydantic_errors_use_public_error_envelope(tmp_path: Path) -> None:
    manager = FakeModelManager(installed=True)
    app = create_app(DwellConfig(home=tmp_path), model_manager=manager)

    with TestClient(app) as client:
        response = client.post(
            "/v1/videos",
            json={"model": "local-video", "prompt": "", "width": 65},
        )
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "invalid_request"
        assert body["error"]["message"] == "The request is invalid."
        assert body["error"]["details"]
        assert app.state.job_store.list() == []

        response = client.post(
            "/v1/videos",
            json={"model": "local-video", "prompt": "aligned", "width": 608},
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
        assert app.state.job_store.list() == []


def test_video_job_is_queued_then_exposed_without_internal_request(tmp_path: Path) -> None:
    manager = FakeModelManager(installed=True)
    app = create_app(DwellConfig(home=tmp_path), model_manager=manager)

    with TestClient(app) as client:
        submitted = client.post(
            "/v1/videos",
            json={"model": "local-video", "prompt": "Local generation"},
        )
        assert submitted.status_code == 202
        assert submitted.json()["status"] == "queued"

        job_id = submitted.json()["id"]
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            response = client.get(f"/v1/jobs/{job_id}")
            if response.json()["status"] == "completed":
                break
            time.sleep(0.01)
        body = response.json()
        assert body["status"] == "completed"
        assert body["progress"] is None
        assert body["output"]["path"] == f"/tmp/{job_id}.mp4"
        assert "request" not in body
        assert manager.generate_calls == 1


def test_job_lookup_and_cancel_errors_use_public_envelope(tmp_path: Path) -> None:
    app = create_app(
        DwellConfig(home=tmp_path),
        model_manager=FakeModelManager(installed=True),
    )

    with TestClient(app) as client:
        response = client.get("/v1/jobs/unknown")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "job_not_found"

        response = client.delete("/v1/jobs/unknown")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "job_not_found"
