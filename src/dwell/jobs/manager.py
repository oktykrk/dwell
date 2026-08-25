from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from dwell.domain import (
    GenerationResult,
    JobError,
    JobOutput,
    JobRecord,
    JobStatus,
    VideoGenerationRequest,
)
from dwell.errors import DwellError
from dwell.jobs.store import JobStore

logger = logging.getLogger(__name__)
_CANCEL_WAIT_SECONDS = 10.0


def _consume_background_task(task: asyncio.Task[Any]) -> None:
    """Retrieve a late cancellation-hook result without abandoning the hook."""

    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("A late runtime cancellation hook failed")


class VideoExecutor(Protocol):
    async def generate_video(
        self,
        request: VideoGenerationRequest,
        job_id: str,
    ) -> GenerationResult | JobOutput: ...


class JobManager:
    """Durable queue with exactly one heavy-inference worker."""

    def __init__(self, store: JobStore, executor: VideoExecutor) -> None:
        self.store = store
        self.executor = executor
        self._wake = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None
        self._generation_task: asyncio.Task[Any] | None = None
        self._active_job_id: str | None = None
        self._stopping = False
        self._cancel_lock = asyncio.Lock()

    @property
    def active_job_id(self) -> str | None:
        return self._active_job_id

    @property
    def running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    async def start(self) -> None:
        if self.running:
            return
        recovered = self.store.recover_interrupted()
        if recovered:
            logger.warning("Recovered %d interrupted inference job(s)", recovered)
        self._stopping = False
        # asyncio synchronization primitives are loop-bound once contended.
        # Recreate them so a stopped manager can safely start on a new loop.
        self._wake = asyncio.Event()
        self._cancel_lock = asyncio.Lock()
        self._worker_task = asyncio.create_task(self._worker(), name="dwell-job-worker")
        self._wake.set()

    async def stop(self) -> None:
        task = self._worker_task
        if task is None:
            return
        self._stopping = True
        active_job_id = self._active_job_id
        if active_job_id is not None:
            try:
                await asyncio.wait_for(self.cancel(active_job_id), timeout=12)
            except TimeoutError:
                logger.error("Timed out while cancelling active job %s", active_job_id)
        self._wake.set()
        try:
            await asyncio.wait_for(task, timeout=15)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            self._worker_task = None
            self._generation_task = None
            self._active_job_id = None
            self.store.close()

    def submit_video(self, request: VideoGenerationRequest) -> JobRecord:
        job = self.store.create_video(request)
        self._wake.set()
        logger.info("Created video job %s for model %s", job.id, job.model)
        return job

    async def cancel(self, job_id: str) -> JobRecord:
        async with self._cancel_lock:
            job = self.store.get(job_id)
            if job.status == JobStatus.QUEUED:
                cancelled = self.store.cancel_queued(job_id)
                if cancelled.status == JobStatus.CANCELLED:
                    self._wake.set()
                    return cancelled
                job = cancelled
            if job.status != JobStatus.RUNNING:
                return job

            generation_task = self._generation_task
            if self._active_job_id != job_id or generation_task is None:
                raise DwellError(
                    "server_busy",
                    f"Job '{job_id}' is dispatching; retry cancellation shortly.",
                    status_code=409,
                )

            hook_task = self._start_cancel_hook(job_id)
            generation_task.cancel()
            deadline = asyncio.get_running_loop().time() + _CANCEL_WAIT_SECONDS
            done, _pending = await asyncio.wait(
                {generation_task},
                timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
            )
            if not done:
                if hook_task is not None:
                    hook_task.add_done_callback(_consume_background_task)
                logger.error("Generation task did not stop after cancellation for job %s", job_id)
                raise DwellError(
                    "server_busy",
                    f"Job '{job_id}' has not confirmed runtime termination.",
                    status_code=409,
                )
            if hook_task is not None:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(hook_task),
                        timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
                    )
                except TimeoutError:
                    hook_task.add_done_callback(_consume_background_task)
                    logger.error("Runtime cancellation hook timed out for job %s", job_id)
                except Exception:
                    logger.exception("Runtime cancellation hook failed for job %s", job_id)
            # The worker owns the state transition and sees the adapter's true
            # termination outcome. Never relabel RUNNING as CANCELLED merely
            # because cancellation was requested.
            for _ in range(100):
                current = self.store.get(job_id)
                if current.status != JobStatus.RUNNING:
                    return current
                await asyncio.sleep(0.01)
            raise DwellError(
                "server_busy",
                f"Cancellation for job '{job_id}' is still being finalized.",
                status_code=409,
            )

    def _start_cancel_hook(self, job_id: str) -> asyncio.Task[Any] | None:
        cancel = getattr(self.executor, "cancel", None)
        if not callable(cancel):
            return None
        try:
            result = cancel(job_id)
        except Exception:
            logger.exception("Runtime cancellation hook failed for job %s", job_id)
            return None
        if not inspect.isawaitable(result):
            return None
        return asyncio.create_task(result, name=f"dwell-cancel-{job_id}")

    async def _worker(self) -> None:
        while not self._stopping:
            job = self.store.claim_next()
            if job is not None:
                await self._execute(job)
                continue

            self._wake.clear()
            # Avoid a missed notification between the empty query and clear().
            if self.store.count(JobStatus.QUEUED):
                self._wake.set()
                continue
            await self._wake.wait()

    async def _execute(self, job: JobRecord) -> None:
        if job.type != "video":
            self.store.fail(
                job.id,
                JobError(
                    code="unsupported_job_type",
                    message=f"Job type '{job.type}' is not supported.",
                ),
            )
            return

        try:
            request = VideoGenerationRequest.model_validate(job.request)
        except Exception as exc:
            self.store.fail(
                job.id,
                JobError(
                    code="invalid_request",
                    message="The persisted job request is invalid.",
                    details={"reason": str(exc)},
                ),
            )
            logger.exception("Persisted request for job %s is invalid", job.id)
            return
        # Another local process may cancel the row after this worker claims it.
        # Re-check before creating any heavyweight runtime process.
        if self.store.get(job.id).status != JobStatus.RUNNING:
            return
        self._active_job_id = job.id
        self._generation_task = asyncio.create_task(
            self.executor.generate_video(request, job.id),
            name=f"dwell-generate-{job.id}",
        )
        logger.info("Started job %s", job.id)
        try:
            result = await self._generation_task
            output = self._output(result)
        except asyncio.CancelledError:
            self.store.mark_cancelled(job.id)
            logger.info("Cancelled job %s", job.id)
        except DwellError as exc:
            self.store.fail(
                job.id,
                JobError(code=exc.code, message=exc.message, details=exc.details),
            )
            logger.warning("Job %s failed: %s", job.id, exc.message)
        except Exception as exc:
            self.store.fail(
                job.id,
                JobError(
                    code="generation_failed",
                    message="Video generation failed.",
                    details={"reason": str(exc)},
                ),
            )
            logger.exception("Job %s failed", job.id)
        else:
            self.store.complete(job.id, output)
            logger.info("Completed job %s: %s", job.id, output.path)
        finally:
            self._generation_task = None
            self._active_job_id = None

    @staticmethod
    def _output(result: Any) -> JobOutput:
        if isinstance(result, JobOutput):
            return result
        if isinstance(result, GenerationResult):
            return JobOutput(
                path=str(result.output_path),
                media_type="video/mp4",
                duration_seconds=result.duration_seconds,
            )
        if isinstance(result, (str, Path)):
            return JobOutput(path=str(result), media_type="video/mp4")
        if isinstance(result, Mapping):
            value = dict(result)
            if "output_path" in value and "path" not in value:
                value["path"] = str(value.pop("output_path"))
            return JobOutput.model_validate(value)
        output_path = getattr(result, "output_path", None)
        if output_path is not None:
            return JobOutput(
                path=str(output_path),
                media_type=getattr(result, "media_type", "video/mp4"),
                duration_seconds=getattr(result, "duration_seconds", None),
            )
        raise TypeError("Video executor returned no output path")


# Kept as a descriptive alias for callers that think in worker terminology.
JobWorker = JobManager
