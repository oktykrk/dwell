from __future__ import annotations

from pathlib import Path

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

    assert [model.id for model in registry.list()] == ["ltx-2.5-bf16", "ltx-2.5-q8"]
    plan = manager.install_plan("ltx-2.5-q8")
    assert plan.downloadable is False
    assert plan.repository is None
    assert plan.already_installed is False
    assert not config.home.exists(), "offline planning must not create operational state"

    with pytest.raises(DwellError, match="no verified install source"):
        manager.install("ltx-2.5-q8")


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
