from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dwell.config import DwellConfig
from dwell.domain import Modality, ModelDefinition, ModelProfile, WeightSource
from dwell.errors import DwellError
from dwell.model_manager import ModelManager
from dwell.registry import ModelRegistry

REVISION = "a" * 40


def _definition(*required: str) -> ModelDefinition:
    return ModelDefinition(
        id="test-video",
        family="test",
        version="1",
        modality=Modality.VIDEO,
        runtime="ltx-2-mlx",
        weights=WeightSource(
            provider="huggingface",
            repository="example/test-video",
            revision=REVISION,
            required_files=required,
        ),
        profile=ModelProfile(quantization="test"),
    )


def _snapshot(config: DwellConfig) -> tuple[Path, Path]:
    repo = config.hf_hub_cache / "models--example--test-video"
    snapshot = repo / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    return repo, snapshot


def test_bundled_registry_is_truthful_about_unconfigured_q8(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    registry = ModelRegistry.load(config)
    manager = ModelManager(config, registry=registry)

    assert [model.id for model in registry.list()] == [
        "ltx-2.5-bf16",
        "ltx-2.5-q8",
        "qwen3-coder-30b-a3b-4bit",
    ]
    plan = manager.install_plan("ltx-2.5-q8")
    assert plan.downloadable is False
    assert plan.repository is None
    assert plan.already_installed is False
    assert not config.home.exists(), "offline planning must not create operational state"

    with pytest.raises(DwellError, match="no verified install source"):
        manager.install("ltx-2.5-q8")


def test_bundled_bf16_install_plan_has_verified_size_and_usage_terms(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    manager = ModelManager(config, registry=ModelRegistry.load(config))

    plan = manager.install_plan("ltx-2.5-bf16")

    assert plan.estimated_size_gb == 71.0
    assert plan.minimum_memory_gb == 48.0
    assert plan.license_url and plan.license_url.endswith("/LICENSE.md")
    assert plan.acceptable_use_url == (
        "https://static.lightricks.com/legal/ltx-acceptable-use-policy.pdf"
    )
    assert "transformer-distilled.safetensors" in plan.required_files
    assert "transformer-dev.safetensors" not in plan.required_files


def test_bundled_qwen_coder_has_a_pinned_downloadable_manifest(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    registry = ModelRegistry.load(config)
    manager = ModelManager(config, registry=registry)

    definition = registry.get("qwen3-coder-30b-a3b-4bit")
    plan = manager.install_plan(definition.id)

    assert definition.modality == Modality.TEXT
    assert definition.runtime == "mlx-lm"
    assert definition.weights.revision == "6e302ea604ad9ab206367e2c501d1571023e7b6d"
    assert "model-*.safetensors" in definition.weights.required_files
    assert plan.repository == "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
    assert plan.estimated_size_gb == 17.2
    assert plan.license_url == "https://www.apache.org/licenses/LICENSE-2.0"
    assert plan.downloadable is True
    assert plan.already_installed is False
    assert not config.home.exists(), "offline planning must not create operational state"


def test_partial_and_zero_byte_files_are_never_installed(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    repo, snapshot = _snapshot(config)
    (snapshot / "config.json").write_bytes(b"")
    (repo / "blobs").mkdir()
    (repo / "blobs" / "weight.incomplete").write_bytes(b"partial")
    registry = ModelRegistry(config, definitions=[_definition("config.json")])

    installation = registry.installation("test-video")
    assert installation.installed is False
    assert installation.partial is True
    assert installation.missing_files == ("config.json",)
    assert installation.incomplete_files


def test_all_weight_shards_must_be_present_and_nonempty(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    _repo, snapshot = _snapshot(config)
    gemma = snapshot / "gemma"
    gemma.mkdir()
    first = gemma / "model-00001-of-00002.safetensors"
    second = gemma / "model-00002-of-00002.safetensors"
    first.write_bytes(b"one")
    registry = ModelRegistry(config, definitions=[_definition("gemma/*.safetensors")])

    assert registry.installation("test-video").installed is False
    second.write_bytes(b"two")
    assert registry.installation("test-video").installed is True
    second.write_bytes(b"")
    assert registry.installation("test-video").installed is False


def test_complete_local_snapshot_resolves_without_network(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    _repo, snapshot = _snapshot(config)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    registry = ModelRegistry(config, definitions=[_definition("config.json")])

    installation = registry.installation("test-video")
    assert installation.installed is True
    assert installation.partial is False
    assert registry.resolve_local_path("test-video") == snapshot.resolve()


def test_explicit_install_pins_hub_and_xet_caches_under_dwell_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    registry = ModelRegistry(config, definitions=[_definition("config.json")])
    manager = ModelManager(config, registry=registry)
    observed: dict[str, str] = {}

    def fake_snapshot_download(**kwargs: object) -> None:
        observed.update(
            {
                "HF_HOME": os.environ["HF_HOME"],
                "HF_HUB_CACHE": os.environ["HF_HUB_CACHE"],
                "HF_XET_CACHE": os.environ["HF_XET_CACHE"],
                "cache_dir": str(kwargs["cache_dir"]),
            }
        )
        _repo, snapshot = _snapshot(config)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setenv("HF_XET_CACHE", str(tmp_path / "outside"))
    previous_home = os.environ.get("HF_HOME")
    previous_hub = os.environ.get("HF_HUB_CACHE")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    result = manager.install("test-video")

    assert result.installed is True
    assert observed == {
        "HF_HOME": str(config.hf_home),
        "HF_HUB_CACHE": str(config.hf_hub_cache),
        "HF_XET_CACHE": str(config.hf_xet_cache),
        "cache_dir": str(config.hf_hub_cache),
    }
    assert os.environ.get("HF_HOME") == previous_home
    assert os.environ.get("HF_HUB_CACHE") == previous_hub
    assert os.environ["HF_XET_CACHE"] == str(tmp_path / "outside")
