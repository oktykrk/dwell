from pathlib import Path

import pytest
from pydantic import ValidationError

from dwell.config import DwellConfig


def test_default_home_is_hidden(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert DwellConfig().home == tmp_path / ".dwell"
    assert DwellConfig.from_env({}).home == tmp_path / ".dwell"


def test_defaults_derive_from_home(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path)

    assert config.api_url == "http://127.0.0.1:8188"
    assert config.models_dir == tmp_path / "models"
    assert config.hf_home == tmp_path / "models" / "huggingface"
    assert config.hf_hub_cache == tmp_path / "models" / "huggingface" / "hub"
    assert config.hf_xet_cache == tmp_path / "models" / "huggingface" / "xet"
    assert config.jobs_db == tmp_path / "state" / "jobs.sqlite"


def test_config_from_env(tmp_path: Path) -> None:
    config = DwellConfig.from_env(
        {
            "DWELL_HOME": str(tmp_path),
            "DWELL_HOST": "127.0.0.1",
            "DWELL_PORT": "19188",
        }
    )

    assert config.home == tmp_path
    assert config.port == 19188


def test_non_loopback_binding_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="only bind to 127.0.0.1"):
        DwellConfig(home=tmp_path, host="0.0.0.0")


@pytest.mark.parametrize("unsafe_home", ["", ".", "/"])
def test_unsafe_home_is_rejected(unsafe_home: str) -> None:
    with pytest.raises(ValidationError):
        DwellConfig(home=unsafe_home)


def test_layout_and_offline_subprocess_environment(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path)
    config.ensure_layout()

    assert (tmp_path / "outputs" / "video").is_dir()
    assert (tmp_path / "tmp" / "ltx-tests").is_dir()
    env = config.subprocess_env(
        {
            "HF_HUB_OFFLINE": "0",
            "PYTHONHOME": "/tmp/host-python",
            "PYTHONPATH": "/tmp/host-modules",
            "TRANSFORMERS_OFFLINE": "0",
            "UV_PROJECT_ENVIRONMENT": "/tmp/host-venv",
            "VIRTUAL_ENV": "/tmp/active-venv",
        }
    )
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_HOME"] == str(config.hf_home)
    assert env["HF_HUB_CACHE"] == str(config.hf_hub_cache)
    assert env["HF_XET_CACHE"] == str(config.hf_xet_cache)
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
    assert "UV_PROJECT_ENVIRONMENT" not in env
    assert "VIRTUAL_ENV" not in env
