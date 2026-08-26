from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from dwell.config import DwellConfig
from dwell.domain import ModelDefinition, ModelState, ModelView
from dwell.errors import DwellError, model_not_found, model_not_installed

_MODEL_LIST_ADAPTER = TypeAdapter(list[ModelDefinition])


@dataclass(frozen=True, slots=True)
class LocalModelInstallation:
    """A network-free inspection of a registered model's local weights."""

    model_id: str
    installed: bool
    partial: bool
    path: Path | None
    missing_files: tuple[str, ...] = ()
    incomplete_files: tuple[Path, ...] = ()

    @property
    def complete(self) -> bool:
        return self.installed


class ModelRegistry:
    """Central model metadata plus deterministic, local-only cache discovery.

    The registry never calls Hugging Face APIs. When ``config/models.json``
    exists under ``DWELL_HOME`` it is treated as an explicit local override;
    otherwise the packaged registry is used.
    """

    def __init__(
        self,
        config: DwellConfig | None = None,
        *,
        path: Path | None = None,
        definitions: Iterable[ModelDefinition | dict[str, Any]] | None = None,
    ) -> None:
        self.config = config or DwellConfig.from_env()
        self.source_path: Path | None = None

        if definitions is None:
            raw, source_path = self._read_registry(path)
            self.source_path = source_path
            parsed = self._parse_document(raw, source_path)
        else:
            try:
                parsed = _MODEL_LIST_ADAPTER.validate_python(list(definitions))
            except ValidationError as exc:
                raise self._invalid_registry("in-memory definitions", exc) from exc

        by_id: dict[str, ModelDefinition] = {}
        for definition in parsed:
            if definition.id in by_id:
                raise self._invalid_registry(
                    str(self.source_path or "in-memory definitions"),
                    ValueError(f"duplicate model id: {definition.id}"),
                )
            self._validate_definition_paths(definition)
            by_id[definition.id] = definition
        self._models = by_id

    @classmethod
    def load(
        cls,
        config: DwellConfig | None = None,
        *,
        path: Path | None = None,
    ) -> ModelRegistry:
        return cls(config, path=path)

    def list(self) -> list[ModelDefinition]:
        return list(self._models.values())

    def list_models(self) -> list[ModelDefinition]:
        """Compatibility alias for consumers that prefer an explicit name."""

        return self.list()

    def get(self, model_id: str) -> ModelDefinition:
        try:
            return self._models[model_id]
        except KeyError:
            raise model_not_found(model_id) from None

    def source_available(self, model_id: str) -> bool:
        source = self.get(model_id).weights
        if source.provider == "huggingface":
            primary = bool(source.repository and source.revision and source.required_files)
            supplemental = all(
                item.repository and item.revision and item.required_files
                for item in source.supplemental_sources
            )
            return primary and supplemental
        if source.provider == "local":
            return bool(source.repository and source.required_files)
        return False

    def installation(self, model_id: str) -> LocalModelInstallation:
        definition = self.get(model_id)
        source = definition.weights
        if source.provider == "unconfigured":
            return LocalModelInstallation(model_id, False, False, None)
        if source.provider == "local":
            return self._inspect_local_source(definition)
        if source.supplemental_sources:
            return self._inspect_composite_source(definition)
        return self._inspect_huggingface_source(definition)

    def resolve_local_path(self, model_id: str) -> Path:
        installation = self.installation(model_id)
        if not installation.installed or installation.path is None:
            raise model_not_installed(model_id)
        # A concrete directory is returned, never a repository ID or branch.
        return installation.path.resolve(strict=True)

    def view(
        self,
        model_id: str,
        *,
        loaded: bool = False,
        runtime_available: bool | None = None,
    ) -> ModelView:
        definition = self.get(model_id)
        installation = self.installation(model_id)
        available = installation.installed
        if runtime_available is not None:
            available = available and runtime_available
        state = (
            ModelState.LOADED
            if loaded
            else (ModelState.INSTALLED if installation.installed else ModelState.NOT_INSTALLED)
        )
        return ModelView(
            id=definition.id,
            family=definition.family,
            version=definition.version,
            modality=definition.modality,
            runtime=definition.runtime,
            quantization=definition.profile.quantization,
            installed=installation.installed,
            available=available,
            loaded=loaded,
            state=state,
            partial=installation.partial,
            cache_path=str(installation.path) if installation.path is not None else None,
        )

    def _read_registry(self, requested_path: Path | None) -> tuple[Any, Path | None]:
        migrate_legacy_q8 = False
        if requested_path is not None:
            source = Path(requested_path).expanduser()
        elif self.config.registry_file.is_file():
            source = self.config.registry_file
            migrate_legacy_q8 = True
        else:
            source = None

        try:
            if source is None:
                text = files("dwell.data").joinpath("models.json").read_text(encoding="utf-8")
            else:
                text = source.read_text(encoding="utf-8")
            raw = json.loads(text)
            if migrate_legacy_q8:
                raw = self._migrate_legacy_q8_placeholder(raw)
            return raw, source
        except (OSError, json.JSONDecodeError) as exc:
            raise self._invalid_registry(str(source or "bundled models.json"), exc) from exc

    @staticmethod
    def _migrate_legacy_q8_placeholder(raw: Any) -> Any:
        if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
            return raw
        models = raw["models"]
        legacy_index = next(
            (
                index
                for index, model in enumerate(models)
                if isinstance(model, dict)
                and model.get("id") == "ltx-2.5-q8"
                and isinstance(model.get("weights"), dict)
                and model["weights"].get("provider") == "unconfigured"
                and model["weights"].get("repository") is None
                and model["weights"].get("revision") is None
                and model["weights"].get("required_files") == []
                and "Registered placeholder only" in str(model["weights"].get("notes", ""))
            ),
            None,
        )
        if legacy_index is None:
            return raw

        bundled = json.loads(
            files("dwell.data").joinpath("models.json").read_text(encoding="utf-8")
        )
        replacement = next(model for model in bundled["models"] if model.get("id") == "ltx-2.5-q8")
        return {**raw, "models": [*models[:legacy_index], replacement, *models[legacy_index + 1 :]]}

    def _parse_document(
        self,
        raw: Any,
        source_path: Path | None,
    ) -> list[ModelDefinition]:
        if isinstance(raw, dict):
            schema_version = raw.get("schema_version", 1)
            if schema_version != 1:
                raise self._invalid_registry(
                    str(source_path or "bundled models.json"),
                    ValueError(f"unsupported schema_version: {schema_version!r}"),
                )
            raw = raw.get("models")
        try:
            return _MODEL_LIST_ADAPTER.validate_python(raw)
        except ValidationError as exc:
            raise self._invalid_registry(str(source_path or "bundled models.json"), exc) from exc

    @staticmethod
    def _invalid_registry(source: str, exc: Exception) -> DwellError:
        return DwellError(
            "invalid_model_registry",
            f"Model registry '{source}' is invalid.",
            details={"reason": str(exc)},
            status_code=500,
        )

    def _validate_definition_paths(self, definition: ModelDefinition) -> None:
        source = definition.weights
        if source.supplemental_sources and source.provider != "huggingface":
            raise self._invalid_registry(
                str(self.source_path or "in-memory definitions"),
                ValueError(
                    f"model '{definition.id}' has supplemental sources without a "
                    "Hugging Face primary source"
                ),
            )
        manifests = [source.required_files]
        manifests.extend(item.required_files for item in source.supplemental_sources)
        for manifest in manifests:
            for required in manifest:
                candidate = Path(required)
                if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
                    raise self._invalid_registry(
                        str(self.source_path or "in-memory definitions"),
                        ValueError(
                            f"model '{definition.id}' has unsafe required file path: {required!r}"
                        ),
                    )

    def _inspect_local_source(self, definition: ModelDefinition) -> LocalModelInstallation:
        repository = definition.weights.repository
        if not repository:
            return LocalModelInstallation(definition.id, False, False, None)
        configured = Path(repository).expanduser()
        path = configured if configured.is_absolute() else self.config.models_dir / configured
        if not path.exists():
            return LocalModelInstallation(definition.id, False, False, None)
        return self._inspect_candidate(definition, path, cache_root=path)

    def _inspect_huggingface_source(
        self,
        definition: ModelDefinition,
    ) -> LocalModelInstallation:
        source = definition.weights
        if not source.repository or not source.revision:
            return LocalModelInstallation(definition.id, False, False, None)

        return self._inspect_huggingface_snapshot(
            definition.id,
            source.repository,
            source.revision,
            source.required_files,
        )

    def _inspect_huggingface_snapshot(
        self,
        model_id: str,
        repository: str,
        revision: str,
        required_files: tuple[str, ...],
    ) -> LocalModelInstallation:
        repo_dir = self.huggingface_repo_dir(repository)
        snapshot = self.huggingface_snapshot_dir(repository, revision)

        if snapshot.is_dir():
            return self._inspect_candidate_files(
                model_id,
                required_files,
                snapshot,
                cache_root=repo_dir,
            )

        incomplete = self._find_incomplete(repo_dir)
        has_cache_state = repo_dir.exists() and self._directory_has_entries(repo_dir)
        return LocalModelInstallation(
            model_id,
            False,
            bool(incomplete or has_cache_state),
            repo_dir if has_cache_state else None,
            tuple(required_files),
            incomplete,
        )

    def huggingface_snapshots(
        self,
        model_id: str,
    ) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
        source = self.get(model_id).weights
        if source.provider != "huggingface" or not source.repository or not source.revision:
            return ()
        primary = ((source.repository, source.revision, source.required_files),)
        supplemental = tuple(
            (item.repository, item.revision, item.required_files)
            for item in source.supplemental_sources
        )
        return primary + supplemental

    def huggingface_snapshot_dir(self, repository: str, revision: str) -> Path:
        repo_dir = self.huggingface_repo_dir(repository)
        snapshot = repo_dir / "snapshots" / revision
        if not snapshot.is_dir():
            snapshot = self._snapshot_from_local_ref(repo_dir, revision) or snapshot
        return snapshot

    def composite_model_dir(self, model_id: str) -> Path:
        snapshots = self.huggingface_snapshots(model_id)
        if len(snapshots) < 2:
            raise ValueError(f"model '{model_id}' is not composite")
        identity = "\0".join(f"{repository}@{revision}" for repository, revision, _ in snapshots)
        digest = sha256(identity.encode("utf-8")).hexdigest()[:24]
        return self.config.composite_models_dir / digest

    def _inspect_composite_source(
        self,
        definition: ModelDefinition,
    ) -> LocalModelInstallation:
        snapshots = self.huggingface_snapshots(definition.id)
        inspected = tuple(
            self._inspect_huggingface_snapshot(definition.id, repository, revision, required)
            for repository, revision, required in snapshots
        )
        composite = self.composite_model_dir(definition.id)
        all_required = tuple(
            required for _repository, _revision, manifest in snapshots for required in manifest
        )
        incomplete = tuple(
            path for installation in inspected for path in installation.incomplete_files
        )
        sources_complete = all(installation.installed for installation in inspected)
        if sources_complete and composite.is_dir():
            candidate = self._inspect_candidate_files(
                definition.id,
                all_required,
                composite,
                cache_root=composite,
            )
            return LocalModelInstallation(
                candidate.model_id,
                candidate.installed,
                candidate.partial,
                candidate.path,
                candidate.missing_files,
                incomplete + candidate.incomplete_files,
            )

        missing = tuple(
            f"{repository}:{required}"
            for (repository, _revision, manifest), installation in zip(
                snapshots, inspected, strict=True
            )
            for required in (
                installation.missing_files or (() if installation.installed else manifest)
            )
        )
        has_state = any(
            installation.partial or installation.installed for installation in inspected
        )
        return LocalModelInstallation(
            definition.id,
            False,
            has_state,
            composite if composite.exists() else None,
            missing or all_required,
            incomplete,
        )

    def huggingface_repo_dir(self, repository: str) -> Path:
        components = repository.split("/")
        if len(components) < 2 or any(part in {"", ".", ".."} for part in components):
            raise self._invalid_registry(
                str(self.source_path or "in-memory definitions"),
                ValueError(f"invalid Hugging Face repository id: {repository!r}"),
            )
        cache_name = "models--" + "--".join(components)
        return self.config.hf_hub_cache / cache_name

    @staticmethod
    def _snapshot_from_local_ref(repo_dir: Path, revision: str) -> Path | None:
        ref = repo_dir / "refs" / revision
        try:
            commit = ref.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not commit or "/" in commit or commit in {".", ".."}:
            return None
        candidate = repo_dir / "snapshots" / commit
        return candidate if candidate.is_dir() else None

    def _inspect_candidate(
        self,
        definition: ModelDefinition,
        candidate: Path,
        *,
        cache_root: Path,
    ) -> LocalModelInstallation:
        return self._inspect_candidate_files(
            definition.id,
            definition.weights.required_files,
            candidate,
            cache_root=cache_root,
        )

    def _inspect_candidate_files(
        self,
        model_id: str,
        required: tuple[str, ...],
        candidate: Path,
        *,
        cache_root: Path,
    ) -> LocalModelInstallation:
        missing = tuple(
            name for name in required if not self._required_file_exists(candidate, name)
        )
        if not required:
            missing = ("<no verification manifest configured>",)
        incomplete = self._find_incomplete(cache_root)
        # Completeness is profile-specific. Optional repository blobs may have
        # interrupted downloads, but they do not invalidate a fully present
        # distilled manifest. Missing required artifacts still make it partial.
        installed = not missing
        has_state = self._directory_has_entries(candidate) or bool(incomplete)
        return LocalModelInstallation(
            model_id,
            installed,
            not installed and has_state,
            candidate,
            missing,
            incomplete,
        )

    @staticmethod
    def _required_file_exists(candidate: Path, pattern: str) -> bool:
        if any(marker in pattern for marker in "*?["):
            try:
                matches = [
                    path
                    for path in candidate.glob(pattern)
                    if path.is_file() and path.stat().st_size > 0
                ]
            except OSError:
                return False
            if not matches:
                return False

            # Hugging Face weight sets may be sharded. One surviving shard is
            # not a complete model, even when no `.incomplete` marker remains.
            shard_pattern = re.compile(r"-(\d+)-of-(\d+)\.safetensors$")
            shard_matches = [shard_pattern.search(path.name) for path in matches]
            shard_matches = [match for match in shard_matches if match is not None]
            if shard_matches:
                totals = {int(match.group(2)) for match in shard_matches}
                if len(totals) != 1:
                    return False
                total = totals.pop()
                present = {int(match.group(1)) for match in shard_matches}
                if present != set(range(1, total + 1)):
                    return False
            return True

        path = candidate / pattern
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    @staticmethod
    def _directory_has_entries(path: Path) -> bool:
        try:
            next(path.iterdir())
        except (OSError, StopIteration):
            return False
        return True

    @staticmethod
    def _find_incomplete(cache_root: Path) -> tuple[Path, ...]:
        if not cache_root.exists():
            return ()
        try:
            return tuple(sorted(cache_root.rglob("*.incomplete")))
        except OSError:
            # An unreadable cache cannot truthfully be called complete.
            return (cache_root / "<unreadable>.incomplete",)
