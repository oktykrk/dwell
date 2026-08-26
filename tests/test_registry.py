from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dwell.config import DwellConfig
from dwell.domain import (
    Modality,
    ModelDefinition,
    ModelProfile,
    SupplementalWeightSource,
    WeightSource,
)
from dwell.model_manager import ModelManager
from dwell.registry import ModelRegistry
from dwell.runtimes.ltx import LTXSubprocessRuntime

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


def test_bundled_q8_has_pinned_composite_sources(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    registry = ModelRegistry.load(config)
    manager = ModelManager(config, registry=registry)

    assert [model.id for model in registry.list()] == [
        "ltx-2.5-bf16",
        "ltx-2.5-q8",
        "qwen3-coder-30b-a3b-4bit",
    ]
    plan = manager.install_plan("ltx-2.5-q8")
    assert plan.downloadable is True
    assert plan.repository == "mlx-community/ltx-2.5-mlx-q8"
    assert plan.revision == "3a80fb22703ef1ca69a89c8f65462b9456b6a361"
    assert plan.supplemental_sources[0].repository == "mlx-community/ltx-2.5-mlx-ditq8"
    assert plan.supplemental_sources[0].revision == ("5724211c8d1600f15062c9c9667440325f252220")
    assert plan.estimated_size_gb == 43.4
    assert plan.minimum_memory_gb == 24.0
    assert plan.already_installed is False
    assert not config.home.exists(), "offline planning must not create operational state"


def test_exact_legacy_q8_placeholder_is_upgraded_in_memory(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    bundled = ModelRegistry.load(config)
    document = {
        "schema_version": 1,
        "models": [definition.model_dump(mode="json") for definition in bundled.list()],
    }
    q8 = next(model for model in document["models"] if model["id"] == "ltx-2.5-q8")
    q8["weights"] = {
        "provider": "unconfigured",
        "repository": None,
        "revision": None,
        "required_files": [],
        "estimated_size_gb": None,
        "notes": "Registered placeholder only. No quantized sibling is published.",
    }
    config.registry_file.parent.mkdir(parents=True)
    config.registry_file.write_text(json.dumps(document), encoding="utf-8")

    migrated = ModelRegistry.load(config)

    assert migrated.source_path == config.registry_file
    assert migrated.get("ltx-2.5-q8").weights.repository == "mlx-community/ltx-2.5-mlx-q8"
    assert '"provider": "unconfigured"' in config.registry_file.read_text(encoding="utf-8")


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


def test_composite_install_downloads_and_assembles_all_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    definition = ModelDefinition(
        id="composite-video",
        family="test",
        version="1",
        modality=Modality.VIDEO,
        runtime="ltx-2-mlx",
        weights=WeightSource(
            provider="huggingface",
            repository="example/components",
            revision="b" * 40,
            required_files=("config.json", "gemma/*.safetensors"),
            supplemental_sources=(
                SupplementalWeightSource(
                    repository="example/transformer",
                    revision="c" * 40,
                    required_files=("transformer-distilled.safetensors",),
                ),
            ),
        ),
        profile=ModelProfile(quantization="q8"),
    )
    registry = ModelRegistry(config, definitions=[definition])
    manager = ModelManager(config, registry=registry)
    calls: list[tuple[str, str, tuple[str, ...]]] = []

    def fake_snapshot_download(**kwargs: object) -> None:
        repository = str(kwargs["repo_id"])
        revision = str(kwargs["revision"])
        required = tuple(kwargs["allow_patterns"])  # type: ignore[arg-type]
        calls.append((repository, revision, required))
        snapshot = registry.huggingface_repo_dir(repository) / "snapshots" / revision
        snapshot.mkdir(parents=True)
        if repository == "example/components":
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "gemma").mkdir()
            (snapshot / "gemma/model.safetensors").write_bytes(b"encoder")
        else:
            (snapshot / "transformer-distilled.safetensors").write_bytes(b"dit")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    result = manager.install(definition.id)

    assert result.installed is True
    assembled = registry.resolve_local_path(definition.id)
    assert assembled == registry.composite_model_dir(definition.id).resolve()
    assert (assembled / "config.json").is_symlink()
    assert (assembled / "gemma/model.safetensors").read_bytes() == b"encoder"
    assert (assembled / "transformer-distilled.safetensors").read_bytes() == b"dit"
    assert calls == [
        ("example/components", "b" * 40, ("config.json", "gemma/*.safetensors")),
        ("example/transformer", "c" * 40, ("transformer-distilled.safetensors",)),
    ]


def test_bundled_q8_composite_matches_ltx_runtime_layout(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    registry = ModelRegistry.load(config)
    manager = ModelManager(config, registry=registry)
    model_id = "ltx-2.5-q8"

    for repository, revision, required_files in registry.huggingface_snapshots(model_id):
        snapshot = registry.huggingface_repo_dir(repository) / "snapshots" / revision
        snapshot.mkdir(parents=True)
        for required in required_files:
            if required == "gemma4-12b-ltx-v1/*.safetensors":
                for shard in range(1, 4):
                    path = snapshot / (f"gemma4-12b-ltx-v1/model-{shard:05d}-of-00003.safetensors")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"weights")
                continue
            if required == "gemma4-12b-ltx-v1/model.safetensors.index.json":
                path = snapshot / required
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    '{"weight_map": {'
                    '"a": "model-00001-of-00003.safetensors", '
                    '"b": "model-00002-of-00003.safetensors", '
                    '"c": "model-00003-of-00003.safetensors"}}',
                    encoding="utf-8",
                )
                continue
            path = snapshot / required
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"data")

    manager._assemble_composite(model_id)
    assembled = registry.resolve_local_path(model_id)
    adapter = LTXSubprocessRuntime(config, registry.resolve_local_path)

    model_path, gemma_path = adapter._validated_model_paths(model_id, assembled)

    assert model_path == assembled
    assert gemma_path == assembled / "gemma4-12b-ltx-v1"
