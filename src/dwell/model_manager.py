from __future__ import annotations

import fcntl
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from dwell.config import DwellConfig
from dwell.domain import (
    GenerationResult,
    JobStatus,
    Modality,
    ModelDefinition,
    ModelView,
    VideoGenerationRequest,
)
from dwell.errors import DwellError, model_not_installed
from dwell.registry import ModelRegistry
from dwell.runtimes.base import TextEngine, TextStream
from dwell.runtimes.registry import RuntimeRegistry

logger = logging.getLogger(__name__)


@contextmanager
def _managed_huggingface_environment(config: DwellConfig) -> Iterator[None]:
    managed = {
        "HF_HOME": str(config.hf_home),
        "HF_HUB_CACHE": str(config.hf_hub_cache),
        "HF_XET_CACHE": str(config.hf_xet_cache),
    }
    previous = {name: os.environ.get(name) for name in managed}
    os.environ.update(managed)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _exclusive_lock(path: Path, message: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DwellError("model_in_use", message, status_code=409) from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class InstallPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_id: str
    provider: str
    repository: str | None
    revision: str | None
    destination: str
    estimated_size_gb: float | None
    license_url: str | None
    acceptable_use_url: str | None
    minimum_memory_gb: float | None
    memory_notes: str | None
    required_files: tuple[str, ...]
    downloadable: bool
    already_installed: bool
    partial: bool
    notes: str | None = None


class ModelManager:
    """Separates local resolution/lifecycle from the one explicit installer."""

    def __init__(
        self,
        config: DwellConfig | None = None,
        *,
        registry: ModelRegistry | None = None,
        runtimes: RuntimeRegistry | None = None,
    ) -> None:
        self.config = config or DwellConfig.from_env()
        self.registry = registry or ModelRegistry.load(self.config)
        self.runtimes = runtimes or RuntimeRegistry(self.config, self.registry)
        self._loaded: set[str] = set()

    def list_models(self) -> list[ModelView]:
        return [self._view(definition) for definition in self.registry.list()]

    def get_model(self, model_id: str) -> ModelView:
        return self._view(self.registry.get(model_id))

    def _view(self, definition: ModelDefinition) -> ModelView:
        return self.registry.view(
            definition.id,
            loaded=definition.id in self._loaded,
            runtime_available=self.runtimes.is_available(definition.runtime),
        )

    def ensure_installed(self, model_id: str) -> ModelView:
        view = self.get_model(model_id)
        if not view.installed:
            raise model_not_installed(model_id)
        return view

    def resolve_local_path(self, model_id: str) -> Path:
        return self.registry.resolve_local_path(model_id)

    def install_plan(self, model_id: str) -> InstallPlan:
        definition = self.registry.get(model_id)
        installation = self.registry.installation(model_id)
        source = definition.weights
        destination = (
            str(self.registry.huggingface_repo_dir(source.repository))
            if source.provider == "huggingface" and source.repository
            else str(self.config.models_dir)
        )
        return InstallPlan(
            model_id=model_id,
            provider=source.provider,
            repository=source.repository,
            revision=source.revision,
            destination=destination,
            estimated_size_gb=source.estimated_size_gb,
            license_url=source.license_url,
            acceptable_use_url=source.acceptable_use_url,
            minimum_memory_gb=definition.profile.minimum_memory_gb,
            memory_notes=definition.profile.memory,
            required_files=source.required_files,
            downloadable=self.registry.source_available(model_id),
            already_installed=installation.installed,
            partial=installation.partial,
            notes=source.notes,
        )

    def install(self, model_id: str, *, dry_run: bool = False) -> InstallPlan | ModelView:
        plan = self.install_plan(model_id)
        if dry_run:
            return plan
        with _exclusive_lock(
            self.config.model_cache_lock_file,
            "Another model cache operation is already in progress.",
        ):
            # Re-plan under the cross-process cache lease. Another explicit
            # installer may have completed while this caller was waiting.
            plan = self.install_plan(model_id)
            if plan.already_installed:
                return self.get_model(model_id)
            if not plan.downloadable or not plan.repository:
                raise DwellError(
                    "invalid_request",
                    f"Model '{model_id}' has no verified install source configured.",
                    details={"notes": plan.notes},
                )

            # This is the only network-capable code path in Dwell. It is reached
            # solely from the explicit install command, never listing or inference.
            # hf-xet reads its chunk-cache location from the process environment
            # when its native session is created. Pin all managed Hugging Face
            # caches before the lazy import so no model data escapes DWELL_HOME.
            with _managed_huggingface_environment(self.config):
                from huggingface_hub import snapshot_download

                self.config.ensure_layout()
                logger.info(
                    "Installing model %s from %s at revision %s",
                    model_id,
                    plan.repository,
                    plan.revision,
                )
                snapshot_download(
                    repo_id=plan.repository,
                    revision=plan.revision,
                    cache_dir=self.config.hf_hub_cache,
                    local_files_only=False,
                    allow_patterns=list(plan.required_files),
                )
            installation = self.registry.installation(model_id)
            if not installation.installed:
                raise DwellError(
                    "model_not_installed",
                    f"Model '{model_id}' download did not pass local completeness checks.",
                    details={
                        "missing_files": list(installation.missing_files),
                        "incomplete_files": [str(path) for path in installation.incomplete_files],
                    },
                )
            logger.info("Installed and verified model %s", model_id)
            return self.get_model(model_id)

    def remove(self, model_id: str) -> ModelView:
        definition = self.registry.get(model_id)
        with _exclusive_lock(
            self.config.start_lock_file,
            "A Dwell service lifecycle operation is in progress; removal was refused.",
        ):
            from dwell.process import (
                _read_ltx_child_state,
                _recorded_ltx_group_may_be_active,
                read_server_state,
            )

            running = read_server_state(self.config)
            if running is not None:
                raise DwellError(
                    "model_in_use",
                    f"Stop Dwell before removing model '{model_id}'.",
                    details={"pid": running.pid},
                    status_code=409,
                )
            child = _read_ltx_child_state(self.config)
            if child is not None and _recorded_ltx_group_may_be_active(child):
                raise DwellError(
                    "model_in_use",
                    f"A recorded LTX child may still be using model '{model_id}'.",
                    details={"pid": child["pid"], "job_id": child["job_id"]},
                    status_code=409,
                )
            with _exclusive_lock(
                self.config.state_dir / f"{definition.runtime}.lock",
                f"Model '{model_id}' may be in use by {definition.runtime}.",
            ):
                with _exclusive_lock(
                    self.config.model_cache_lock_file,
                    "Another model cache operation is already in progress.",
                ):
                    return self._remove_locked(definition)

    def _remove_locked(self, definition: ModelDefinition) -> ModelView:
        model_id = definition.id
        installation = self.registry.installation(model_id)
        if model_id in self._loaded or self._has_active_job(model_id):
            raise DwellError(
                "model_in_use",
                f"Model '{model_id}' is being used and cannot be removed.",
                status_code=409,
            )
        if not installation.installed and not installation.partial:
            return self.get_model(model_id)
        if definition.weights.provider == "local":
            raise DwellError(
                "invalid_request",
                "Dwell will not delete externally configured local model paths.",
            )
        if definition.weights.provider != "huggingface":
            raise DwellError(
                "invalid_request",
                f"Model '{model_id}' has no removable managed cache entry.",
            )
        if installation.partial:
            raise DwellError(
                "invalid_request",
                "Conservative removal refused: partial shared-cache blobs cannot be mapped safely.",
                details={"cache_path": str(installation.path) if installation.path else None},
            )

        self._ensure_repository_not_shared(definition)
        commit_hash = installation.path.name if installation.path is not None else ""
        from huggingface_hub import scan_cache_dir

        cache = scan_cache_dir(self.config.hf_hub_cache)
        matching = [
            revision
            for repo in cache.repos
            if repo.repo_type == "model" and repo.repo_id == definition.weights.repository
            for revision in repo.revisions
            if revision.commit_hash == commit_hash
        ]
        if len(matching) != 1:
            raise DwellError(
                "invalid_request",
                (
                    "Conservative removal refused: the exact cache revision "
                    "was not uniquely identified."
                ),
            )
        strategy = cache.delete_revisions(commit_hash)
        logger.info(
            "Removing model %s revision %s (%s)",
            model_id,
            commit_hash,
            strategy.expected_freed_size_str,
        )
        strategy.execute()
        return self.get_model(model_id)

    def _ensure_repository_not_shared(self, definition: ModelDefinition) -> None:
        for other in self.registry.list():
            if other.id == definition.id:
                continue
            if (
                other.weights.provider == "huggingface"
                and other.weights.repository == definition.weights.repository
                and (
                    self.registry.installation(other.id).installed
                    or self.registry.installation(other.id).partial
                )
            ):
                raise DwellError(
                    "model_in_use",
                    f"Cache repository is also used by model '{other.id}'.",
                    status_code=409,
                )

    def _has_active_job(self, model_id: str) -> bool:
        if not self.config.jobs_db.is_file():
            return False
        try:
            with sqlite3.connect(self.config.jobs_db) as connection:
                row = connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE model = ? AND status IN (?, ?)",
                    (model_id, JobStatus.QUEUED.value, JobStatus.RUNNING.value),
                ).fetchone()
        except sqlite3.Error:
            # An unreadable state database is not a safe basis for deletion.
            return True
        return bool(row and row[0])

    async def load(self, model_id: str) -> ModelView:
        self.ensure_installed(model_id)
        definition = self.registry.get(model_id)
        runtime = self.runtimes.get(definition.runtime)
        model_path = self.resolve_local_path(model_id)
        await self.runtimes.acquire_inference(wait=False)
        try:
            released = await self.runtimes.release_resident_except(definition.runtime)
            self._loaded.difference_update(released)
            await runtime.prepare(definition, model_path)
        finally:
            await self.runtimes.release_inference()
        if runtime.capabilities.persistent_loading:
            self._loaded.add(model_id)
            logger.info("Loaded model %s", model_id)
        else:
            logger.info("Model %s is ready for on-demand inference", model_id)
        return self.get_model(model_id)

    async def unload(self, model_id: str) -> ModelView:
        definition = self.registry.get(model_id)
        runtime = self.runtimes.get(definition.runtime)
        if runtime.capabilities.persistent_loading and model_id in self._loaded:
            await self.runtimes.acquire_inference(wait=False)
            try:
                await runtime.release(definition)
            finally:
                await self.runtimes.release_inference()
            self._loaded.discard(model_id)
            logger.info("Unloaded model %s", model_id)
        return self.get_model(model_id)

    async def unload_all(self) -> list[ModelView]:
        await self.runtimes.acquire_inference(wait=False)
        try:
            await self.runtimes.release_all()
            self._loaded.clear()
        finally:
            await self.runtimes.release_inference()
        return self.list_models()

    async def generate_video(
        self,
        request: VideoGenerationRequest,
        job_id: str,
        cancel_event: Any = None,
    ) -> GenerationResult:
        view = self.ensure_installed(request.model)
        if view.modality != Modality.VIDEO:
            raise DwellError(
                "invalid_request",
                f"Model '{request.model}' does not support video generation.",
                status_code=422,
            )
        definition = self.registry.get(request.model)
        engine = self.runtimes.video_engine(definition)
        await self.runtimes.acquire_inference(wait=True)
        try:
            released = await self.runtimes.release_resident_except(definition.runtime)
            self._loaded.difference_update(released)
            return await engine.generate(request, job_id, cancel_event)
        finally:
            await self.runtimes.release_inference()

    def _text_engine(self, model_id: str) -> tuple[ModelDefinition, TextEngine]:
        view = self.ensure_installed(model_id)
        if view.modality != Modality.TEXT:
            raise DwellError(
                "invalid_request",
                f"Model '{model_id}' does not support text generation.",
                status_code=422,
            )
        if not view.available:
            raise DwellError(
                "runtime_not_available",
                f"Runtime '{view.runtime}' is not available locally.",
                status_code=503,
            )
        definition = self.registry.get(model_id)
        return definition, self.runtimes.text_engine(definition)

    async def complete_text(
        self,
        model_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        definition, engine = self._text_engine(model_id)
        await self.runtimes.acquire_inference(wait=False)
        try:
            released = await self.runtimes.release_resident_except(definition.runtime)
            self._loaded.difference_update(released)
            result = await engine.complete(definition, payload)
            if engine.capabilities.persistent_loading:
                self._loaded.add(model_id)
            return result
        finally:
            await self.runtimes.release_inference()

    async def open_text_stream(
        self,
        model_id: str,
        payload: dict[str, Any],
    ) -> TextStream:
        definition, engine = self._text_engine(model_id)
        await self.runtimes.acquire_inference(wait=False)
        try:
            released = await self.runtimes.release_resident_except(definition.runtime)
            self._loaded.difference_update(released)
            stream = await engine.open_stream(definition, payload)
        except BaseException:
            await self.runtimes.release_inference()
            raise
        if engine.capabilities.persistent_loading:
            self._loaded.add(model_id)
        return TextStream(stream, self.runtimes.release_inference)

    async def cancel(self, job_id: str) -> bool:
        cancelled = False
        for runtime in list(self.runtimes._instances.values()):
            cancel = getattr(runtime, "cancel", None)
            if cancel is not None and await cancel(job_id):
                cancelled = True
        return cancelled

    async def shutdown(self) -> None:
        await self.runtimes.release_all()
        self._loaded.clear()
