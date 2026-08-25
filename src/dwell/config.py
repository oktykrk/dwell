from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DwellConfig(BaseModel):
    """Resolved local configuration. No property in this class contacts the network."""

    model_config = ConfigDict(frozen=True)

    home: Path = Field(default_factory=lambda: Path.home() / ".dwell")
    host: str = "127.0.0.1"
    port: int = Field(default=8188, ge=1, le=65535)

    @field_validator("home", mode="before")
    @classmethod
    def expand_home(cls, value: object) -> Path:
        raw = str(value).strip()
        if not raw:
            raise ValueError("DWELL_HOME must not be empty")
        expanded = Path(raw).expanduser()
        if not expanded.is_absolute():
            raise ValueError("DWELL_HOME must be an absolute path")
        path = expanded.resolve(strict=False)
        if path == Path(path.anchor):
            raise ValueError("DWELL_HOME must not be a filesystem root")
        return path

    @field_validator("host")
    @classmethod
    def localhost_only(cls, value: str) -> str:
        if value != "127.0.0.1":
            raise ValueError("Dwell may only bind to 127.0.0.1")
        return value

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> DwellConfig:
        values = os.environ if env is None else env
        return cls(
            home=values.get("DWELL_HOME", str(Path.home() / ".dwell")),
            host=values.get("DWELL_HOST", "127.0.0.1"),
            port=int(values.get("DWELL_PORT", "8188")),
        )

    @property
    def api_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def runtimes_dir(self) -> Path:
        return self.home / "runtimes"

    @property
    def services_dir(self) -> Path:
        return self.home / "services"

    @property
    def outputs_dir(self) -> Path:
        return self.home / "outputs"

    @property
    def state_dir(self) -> Path:
        return self.home / "state"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def tmp_dir(self) -> Path:
        return self.home / "tmp"

    @property
    def config_dir(self) -> Path:
        return self.home / "config"

    @property
    def bin_dir(self) -> Path:
        return self.home / "bin"

    @property
    def hf_home(self) -> Path:
        return self.models_dir / "huggingface"

    @property
    def hf_hub_cache(self) -> Path:
        return self.hf_home / "hub"

    @property
    def hf_xet_cache(self) -> Path:
        return self.hf_home / "xet"

    @property
    def registry_file(self) -> Path:
        return self.config_dir / "models.json"

    @property
    def pid_file(self) -> Path:
        return self.state_dir / "server.pid"

    @property
    def server_state_file(self) -> Path:
        return self.state_dir / "server.json"

    @property
    def start_lock_file(self) -> Path:
        return self.state_dir / "server.start.lock"

    @property
    def ltx_active_process_file(self) -> Path:
        return self.state_dir / "ltx-2-mlx.active.json"

    @property
    def ltx_lock_file(self) -> Path:
        return self.state_dir / "ltx-2-mlx.lock"

    @property
    def model_cache_lock_file(self) -> Path:
        return self.state_dir / "model-cache.lock"

    @property
    def setup_lock_file(self) -> Path:
        return self.state_dir / "setup.lock"

    @property
    def setup_state_file(self) -> Path:
        return self.state_dir / "setup.json"

    @property
    def jobs_db(self) -> Path:
        return self.state_dir / "jobs.sqlite"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "dwell.log"

    def ensure_layout(self) -> None:
        directories = (
            self.models_dir,
            self.hf_home,
            self.hf_hub_cache,
            self.hf_xet_cache,
            self.runtimes_dir,
            self.services_dir / "api",
            self.outputs_dir / "video",
            self.outputs_dir / "image",
            self.outputs_dir / "audio",
            self.outputs_dir / "text",
            self.state_dir,
            self.logs_dir,
            self.tmp_dir / "ltx-tests",
            self.config_dir,
            self.bin_dir,
        )
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def subprocess_env(self, base: Mapping[str, str] | None = None) -> dict[str, str]:
        """Environment used for inference and diagnostics; always local-only."""

        result = dict(os.environ if base is None else base)
        for name in tuple(result):
            if name.startswith("UV_") or name in {
                "PYTHONHOME",
                "PYTHONPATH",
                "VIRTUAL_ENV",
            }:
                result.pop(name)
        result.update(
            {
                "DWELL_HOME": str(self.home),
                "DWELL_HOST": self.host,
                "DWELL_PORT": str(self.port),
                "HF_HOME": str(self.hf_home),
                "HF_HUB_CACHE": str(self.hf_hub_cache),
                "HF_XET_CACHE": str(self.hf_xet_cache),
                "HF_HUB_OFFLINE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "UV_OFFLINE": "1",
                "UV_NO_SYNC": "1",
                "UV_PYTHON_DOWNLOADS": "never",
            }
        )
        return result

    def display(self) -> dict[str, str | int]:
        return {
            "home": str(self.home),
            "api_host": self.host,
            "api_port": self.port,
            "models_dir": str(self.models_dir),
            "runtimes_dir": str(self.runtimes_dir),
            "outputs_dir": str(self.outputs_dir),
            "logs_dir": str(self.logs_dir),
            "state_dir": str(self.state_dir),
            "HF_HOME": str(self.hf_home),
            "HF_HUB_CACHE": str(self.hf_hub_cache),
            "HF_XET_CACHE": str(self.hf_xet_cache),
        }
