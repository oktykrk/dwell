from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from dwell.domain import (
    TERMINAL_JOB_STATUSES,
    JobError,
    JobOutput,
    JobRecord,
    JobStatus,
    VideoGenerationRequest,
    utc_now,
)
from dwell.errors import job_not_found

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    request_json TEXT NOT NULL,
    error_json TEXT,
    output_json TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status_created_idx
    ON jobs(status, created_at);
"""


def _timestamp(value: datetime | None = None) -> str:
    return (value or utc_now()).isoformat()


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


class JobStore:
    """Small, thread-safe SQLite repository for durable inference jobs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA)

    @staticmethod
    def _record(row: sqlite3.Row) -> JobRecord:
        error = json.loads(row["error_json"]) if row["error_json"] else None
        output = json.loads(row["output_json"]) if row["output_json"] else None
        return JobRecord(
            id=row["id"],
            type=row["type"],
            model=row["model"],
            status=JobStatus(row["status"]),
            progress=row["progress"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=(datetime.fromisoformat(row["started_at"]) if row["started_at"] else None),
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
            request=json.loads(row["request_json"]),
            error=JobError.model_validate(error) if error is not None else None,
            output=JobOutput.model_validate(output) if output is not None else None,
        )

    def create_video(
        self,
        request: VideoGenerationRequest,
        *,
        job_id: str | None = None,
    ) -> JobRecord:
        return self.create(
            job_type="video",
            model=request.model,
            request=request.model_dump(mode="json"),
            job_id=job_id,
        )

    def create(
        self,
        *,
        job_type: str,
        model: str,
        request: dict[str, Any],
        job_id: str | None = None,
    ) -> JobRecord:
        identifier = job_id or str(uuid.uuid4())
        created_at = _timestamp()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, type, model, status, progress, created_at,
                    started_at, completed_at, request_json, error_json, output_json
                ) VALUES (?, ?, ?, ?, NULL, ?, NULL, NULL, ?, NULL, NULL)
                """,
                (
                    identifier,
                    job_type,
                    model,
                    JobStatus.QUEUED.value,
                    created_at,
                    _json(request),
                ),
            )
        return self.get(identifier)

    def get(self, job_id: str) -> JobRecord:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise job_not_found(job_id)
        return self._record(row)

    def list(
        self,
        *,
        statuses: Iterable[JobStatus | str] | None = None,
        limit: int | None = None,
    ) -> list[JobRecord]:
        parameters: list[Any] = []
        where = ""
        if statuses is not None:
            status_values = [JobStatus(status).value for status in statuses]
            if not status_values:
                return []
            placeholders = ",".join("?" for _ in status_values)
            where = f" WHERE status IN ({placeholders})"
            parameters.extend(status_values)
        limit_clause = ""
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            limit_clause = " LIMIT ?"
            parameters.append(limit)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM jobs{where} ORDER BY created_at DESC{limit_clause}",  # noqa: S608
                parameters,
            ).fetchall()
        return [self._record(row) for row in rows]

    def count(self, status: JobStatus | str) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE status = ?",
                (JobStatus(status).value,),
            ).fetchone()
        return int(row["count"])

    def active(self) -> JobRecord | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE status = ? ORDER BY started_at LIMIT 1",
                (JobStatus.RUNNING.value,),
            ).fetchone()
        return self._record(row) if row is not None else None

    def claim_next(self) -> JobRecord | None:
        """Atomically move the oldest queued job to running."""

        started_at = _timestamp()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id FROM jobs WHERE status = ? ORDER BY created_at, id LIMIT 1",
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if row is None:
                return None
            job_id = row["id"]
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = ?, completed_at = NULL,
                    error_json = NULL, output_json = NULL
                WHERE id = ? AND status = ?
                """,
                (
                    JobStatus.RUNNING.value,
                    started_at,
                    job_id,
                    JobStatus.QUEUED.value,
                ),
            )
            if cursor.rowcount != 1:
                return None
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        return self._record(claimed)

    def update_progress(self, job_id: str, progress: float | None) -> JobRecord:
        if progress is not None and not 0 <= progress <= 1:
            raise ValueError("progress must be between zero and one")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET progress = ? WHERE id = ? AND status = ?",
                (progress, job_id, JobStatus.RUNNING.value),
            )
        if cursor.rowcount == 0:
            return self.get(job_id)
        return self.get(job_id)

    def complete(self, job_id: str, output: JobOutput) -> JobRecord:
        return self._finish(
            job_id,
            status=JobStatus.COMPLETED,
            output_json=_json(output),
        )

    def fail(self, job_id: str, error: JobError) -> JobRecord:
        return self._finish(
            job_id,
            status=JobStatus.FAILED,
            error_json=_json(error),
        )

    def mark_cancelled(self, job_id: str) -> JobRecord:
        error = JobError(code="job_cancelled", message="The job was cancelled.")
        completed_at = _timestamp()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = ?, completed_at = ?, error_json = ?, output_json = NULL
                WHERE id = ? AND status IN (?, ?)
                """,
                (
                    JobStatus.CANCELLED.value,
                    completed_at,
                    _json(error),
                    job_id,
                    JobStatus.QUEUED.value,
                    JobStatus.RUNNING.value,
                ),
            )
        if cursor.rowcount == 0:
            return self.get(job_id)
        return self.get(job_id)

    def cancel_queued(self, job_id: str) -> JobRecord:
        """Atomically cancel only a job that has not been claimed."""

        error = JobError(code="job_cancelled", message="The job was cancelled.")
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, completed_at = ?, error_json = ?, output_json = NULL
                WHERE id = ? AND status = ?
                """,
                (
                    JobStatus.CANCELLED.value,
                    _timestamp(),
                    _json(error),
                    job_id,
                    JobStatus.QUEUED.value,
                ),
            )
        return self.get(job_id)

    def _finish(
        self,
        job_id: str,
        *,
        status: JobStatus,
        error_json: str | None = None,
        output_json: str | None = None,
    ) -> JobRecord:
        completed_at = _timestamp()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = ?, completed_at = ?, error_json = ?, output_json = ?
                WHERE id = ? AND status = ?
                """,
                (
                    status.value,
                    completed_at,
                    error_json,
                    output_json,
                    job_id,
                    JobStatus.RUNNING.value,
                ),
            )
        if cursor.rowcount == 0:
            return self.get(job_id)
        return self.get(job_id)

    def recover_interrupted(self) -> int:
        """Fail work left running by a prior server; it is not safe to replay it."""

        error = JobError(
            code="generation_interrupted",
            message="Generation was interrupted before the previous server stopped.",
        )
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET status = ?, completed_at = ?, error_json = ?, output_json = NULL
                WHERE status = ?
                """,
                (
                    JobStatus.FAILED.value,
                    _timestamp(),
                    _json(error),
                    JobStatus.RUNNING.value,
                ),
            )
        return cursor.rowcount

    def clear_terminal(self) -> int:
        values = tuple(status.value for status in TERMINAL_JOB_STATUSES)
        placeholders = ",".join("?" for _ in values)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"DELETE FROM jobs WHERE status IN ({placeholders})",  # noqa: S608
                values,
            )
        return cursor.rowcount

    def close(self) -> None:
        """Connections are short-lived; retained for a uniform repository lifecycle."""
