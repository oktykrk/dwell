from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class Modality(StrEnum):
    TEXT = "text"
    STRUCTURED_JSON = "structured_json"
    EMBEDDINGS = "embeddings"
    IMAGE = "image"
    VIDEO = "video"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    AUDIO = "audio"


class ModelState(StrEnum):
    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    ERROR = "error"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_JOB_STATUSES = {
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
}


class RuntimeCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    persistent_loading: bool = False
    progress_reporting: bool = False
    cancellation: bool = True
    streaming: bool = False
    structured_output: bool = False


class WeightSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Literal["huggingface", "unconfigured", "local"]
    repository: str | None = None
    revision: str | None = None
    required_files: tuple[str, ...] = ()
    estimated_size_gb: float | None = None
    license_url: str | None = None
    acceptable_use_url: str | None = None
    notes: str | None = None


class ModelProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    quantization: str | None = None
    disk: str | None = None
    memory: str | None = None
    minimum_memory_gb: float | None = Field(default=None, gt=0)


class ModelDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    family: str
    version: str
    modality: Modality
    runtime: str
    weights: WeightSource
    profile: ModelProfile
    runtime_requirements: tuple[str, ...] = ()
    capabilities: RuntimeCapabilities = Field(default_factory=RuntimeCapabilities)


class ModelView(BaseModel):
    id: str
    family: str
    version: str
    modality: Modality
    runtime: str
    quantization: str | None
    registered: bool = True
    installed: bool
    available: bool
    loaded: bool
    state: ModelState
    partial: bool = False
    cache_path: str | None = None


class VideoGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str = Field(min_length=1, max_length=20_000)
    width: int = Field(default=576, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    frames: int = Field(default=121, ge=1, le=2001)
    fps: float = Field(default=24, gt=0, le=240)
    seed: int | None = Field(default=None, ge=0, le=2**31 - 1)

    @field_validator("width", "height")
    @classmethod
    def dimensions_are_aligned(cls, value: int) -> int:
        if value % 64:
            raise ValueError("video dimensions must be divisible by 64")
        return value

    @field_validator("frames")
    @classmethod
    def frames_match_latent_stride(cls, value: int) -> int:
        if (value - 1) % 8:
            raise ValueError("frames must be 8n+1 for the LTX runtime")
        return value


class JobError(BaseModel):
    code: str
    message: str
    details: Any = None


class JobOutput(BaseModel):
    path: str
    media_type: str | None = None
    duration_seconds: float | None = None


class JobRecord(BaseModel):
    id: str
    type: str
    model: str
    status: JobStatus
    progress: float | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: JobError | None = None
    output: JobOutput | None = None
    request: dict[str, Any] = Field(default_factory=dict, exclude=True)


class GenerationResult(BaseModel):
    output_path: Path
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
