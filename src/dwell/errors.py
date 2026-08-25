from __future__ import annotations

from typing import Any


class DwellError(Exception):
    """An expected operational error with a stable public representation."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Any = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
        self.status_code = status_code

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }


def model_not_found(model_id: str) -> DwellError:
    return DwellError(
        "model_not_found",
        f"Model '{model_id}' is not registered.",
        status_code=404,
    )


def model_not_installed(model_id: str) -> DwellError:
    return DwellError(
        "model_not_installed",
        f"Model '{model_id}' is not installed locally.",
        status_code=409,
    )


def job_not_found(job_id: str) -> DwellError:
    return DwellError(
        "job_not_found",
        f"Job '{job_id}' was not found.",
        status_code=404,
    )
