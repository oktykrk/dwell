from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dwell.domain import (
    GenerationResult,
    JobOutput,
    JobStatus,
    VideoGenerationRequest,
)
from dwell.errors import DwellError
from dwell.jobs import JobManager, JobStore
from dwell.jobs import manager as manager_module


def video_request(model: str = "local-video") -> VideoGenerationRequest:
    return VideoGenerationRequest(model=model, prompt="Rain over Istanbul")


def test_job_store_persists_request_timestamps_output_and_errors(tmp_path: Path) -> None:
    database = tmp_path / "state" / "jobs.sqlite"
    store = JobStore(database)
    queued = store.create_video(video_request(), job_id="job-one")

    assert queued.status == JobStatus.QUEUED
    assert queued.progress is None
    assert queued.request["prompt"] == "Rain over Istanbul"
    assert queued.started_at is None

    running = store.claim_next()
    assert running is not None
    assert running.id == queued.id
    assert running.status == JobStatus.RUNNING
    assert running.started_at is not None

    completed = store.complete(
        queued.id,
        JobOutput(path="/tmp/job-one.mp4", media_type="video/mp4", duration_seconds=1.5),
    )
    assert completed.status == JobStatus.COMPLETED
    assert completed.completed_at is not None
    assert completed.output is not None
    assert completed.output.path == "/tmp/job-one.mp4"

    reopened = JobStore(database).get(queued.id)
    assert reopened.request == queued.request
    assert reopened.output == completed.output
    assert reopened.created_at == queued.created_at


def test_job_store_recovers_interrupted_running_work_truthfully(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    queued = store.create_video(video_request(), job_id="interrupted")
    store.claim_next()

    assert store.recover_interrupted() == 1
    recovered = store.get(queued.id)
    assert recovered.status == JobStatus.FAILED
    assert recovered.completed_at is not None
    assert recovered.error is not None
    assert recovered.error.code == "generation_interrupted"


class ControlledExecutor:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started: asyncio.Queue[str] = asyncio.Queue()
        self.active = 0
        self.maximum_active = 0
        self.cancelled: set[str] = set()

    async def generate_video(
        self,
        _request: VideoGenerationRequest,
        job_id: str,
    ) -> GenerationResult:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await self.started.put(job_id)
        try:
            await self.release.wait()
            return GenerationResult(
                output_path=Path(f"/tmp/{job_id}.mp4"),
                duration_seconds=0.1,
            )
        except asyncio.CancelledError:
            self.cancelled.add(job_id)
            raise
        finally:
            self.active -= 1


@pytest.mark.asyncio
async def test_job_manager_runs_one_heavy_job_and_cancels_runtime(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    executor = ControlledExecutor()
    jobs = JobManager(store, executor)
    await jobs.start()
    try:
        first = jobs.submit_video(video_request())
        second = jobs.submit_video(video_request())
        assert await asyncio.wait_for(executor.started.get(), 1) == first.id
        await asyncio.sleep(0)
        assert store.get(second.id).status == JobStatus.QUEUED
        assert executor.maximum_active == 1

        cancelled = await jobs.cancel(first.id)
        assert cancelled.status == JobStatus.CANCELLED
        assert first.id in executor.cancelled

        assert await asyncio.wait_for(executor.started.get(), 1) == second.id
        executor.release.set()
        for _ in range(100):
            if store.get(second.id).status == JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.01)
        assert store.get(second.id).status == JobStatus.COMPLETED
        assert executor.maximum_active == 1
    finally:
        await jobs.stop()


def test_cancel_queued_never_relabels_a_claimed_job(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_video(video_request(), job_id="claim-race")
    claimed = store.claim_next()

    assert claimed is not None
    assert claimed.id == job.id
    assert store.cancel_queued(job.id).status == JobStatus.RUNNING


@pytest.mark.asyncio
async def test_running_job_without_local_dispatch_cannot_be_falsely_cancelled(
    tmp_path: Path,
) -> None:
    store = JobStore(tmp_path / "jobs.sqlite")
    job = store.create_video(video_request(), job_id="other-process")
    assert store.claim_next() is not None
    jobs = JobManager(store, ControlledExecutor())

    with pytest.raises(DwellError) as error:
        await jobs.cancel(job.id)

    assert error.value.code == "server_busy"
    assert store.get(job.id).status == JobStatus.RUNNING


class StubbornExecutor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate_video(
        self,
        _request: VideoGenerationRequest,
        job_id: str,
    ) -> GenerationResult:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            # Simulate a broken third-party runtime that ignores task
            # cancellation. Dwell must not falsely persist `cancelled`.
            await self.release.wait()
        return GenerationResult(
            output_path=Path(f"/tmp/{job_id}.mp4"),
            duration_seconds=0.1,
        )


@pytest.mark.asyncio
async def test_cancel_is_bounded_and_truthful_when_executor_will_not_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "_CANCEL_WAIT_SECONDS", 0.01)
    store = JobStore(tmp_path / "jobs.sqlite")
    executor = StubbornExecutor()
    jobs = JobManager(store, executor)
    await jobs.start()
    job = jobs.submit_video(video_request())
    await asyncio.wait_for(executor.started.wait(), 1)
    try:
        with pytest.raises(DwellError) as error:
            await jobs.cancel(job.id)
        assert error.value.code == "server_busy"
        assert store.get(job.id).status == JobStatus.RUNNING
    finally:
        executor.release.set()
        await jobs.stop()
