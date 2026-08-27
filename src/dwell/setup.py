from __future__ import annotations

import configparser
import fcntl
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, field_validator, model_validator

from dwell import __version__
from dwell.config import DwellConfig
from dwell.doctor import DoctorCheck, has_errors, run_doctor
from dwell.errors import DwellError

_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TOOL_VERSION_ARGUMENTS = {
    "uv": ("--version",),
    "ffmpeg": ("-version",),
    "ffprobe": ("-version",),
}
_STRUCTURAL_FILE_LIMIT = 64 * 1024
_VENV_PROVENANCE_FILE = ".dwell-provenance.json"
_SOURCE_ARCHIVE_FILE = ".dwell-source.tar.gz"
_SOURCE_MANIFEST_FILE = ".dwell-source.json"
_SOURCE_MANIFEST_LIMIT = 2 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_MAX_ARCHIVE_CONTENT_BYTES = 512 * 1024 * 1024
_MAX_ARCHIVE_ENTRIES = 20_000
_ARCHIVE_SOURCE_FIELDS = frozenset({"id", "archive_url", "archive_sha256", "commit", "python"})


class SetupMode(StrEnum):
    INSTALL = "install"
    CHECK = "check"
    REPAIR = "repair"
    UPGRADE = "upgrade"


class RuntimeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    repository: str | None = None
    branch: str | None = None
    archive_url: str | None = None
    archive_sha256: str | None = None
    commit: str
    python: str

    @field_validator("id", "python")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("repository", "branch", "archive_url")
    @classmethod
    def optional_nonempty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("commit")
    @classmethod
    def immutable_commit(cls, value: str) -> str:
        if not _COMMIT_RE.fullmatch(value):
            raise ValueError("must be a full 40-character lowercase Git commit")
        return value

    @field_validator("archive_sha256")
    @classmethod
    def immutable_archive(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError("must be a 64-character lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def complete_source(self) -> RuntimeSpec:
        if (self.repository is None) != (self.branch is None):
            raise ValueError("repository and branch must be provided together")
        if (self.archive_url is None) != (self.archive_sha256 is None):
            raise ValueError("archive_url and archive_sha256 must be provided together")
        if self.repository is None and self.archive_url is None:
            raise ValueError("a Git repository or pinned archive must be provided")
        if self.archive_url is not None:
            parsed = urllib.parse.urlsplit(self.archive_url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ValueError("archive_url must be an absolute HTTPS URL")
            if self.commit not in self.archive_url:
                raise ValueError("archive_url must contain the full pinned commit")
        return self

    @property
    def uses_archive(self) -> bool:
        return self.archive_url is not None


_RUNTIME_LIST = TypeAdapter(list[RuntimeSpec])


@dataclass(frozen=True, slots=True)
class RuntimeInspection:
    exists: bool
    source_valid: bool
    remote: str | None
    commit: str | None
    source_hash: str | None
    dirty: bool
    venv_exists: bool
    venv_healthy: bool
    problems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SetupReport:
    mode: SetupMode
    changed: bool
    healthy: bool
    messages: tuple[str, ...]
    doctor_checks: tuple[DoctorCheck, ...] = ()


@dataclass(frozen=True, slots=True)
class MutationOutcome:
    changed: bool
    messages: tuple[str, ...]
    backup: Path | None = None
    previous_inspection: RuntimeInspection | None = None
    preserve_backup: bool = False


Runner = Callable[..., subprocess.CompletedProcess[str]]
DoctorRunner = Callable[[DwellConfig], list[DoctorCheck]]
ArchiveDownloader = Callable[[str, Path, str], None]


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 60,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def _download_archive(url: str, destination: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    downloaded = 0
    created = False
    try:
        with httpx.stream(
            "GET",
            url,
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": f"dwell/{__version__}",
            },
            follow_redirects=True,
            timeout=60,
        ) as response:
            response.raise_for_status()
            if any(item.url.scheme != "https" for item in (*response.history, response)):
                raise DwellError("setup_failed", "Runtime archive redirected outside HTTPS.")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > _MAX_ARCHIVE_BYTES:
                raise DwellError("setup_failed", "Runtime archive exceeds the download limit.")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(destination, flags, 0o600)
            created = True
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as output:
                    for chunk in response.iter_raw(chunk_size=1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > _MAX_ARCHIVE_BYTES:
                            raise DwellError(
                                "setup_failed", "Runtime archive exceeds the download limit."
                            )
                        digest.update(chunk)
                        output.write(chunk)
            finally:
                os.close(descriptor)
    except DwellError:
        if created:
            destination.unlink(missing_ok=True)
        raise
    except (OSError, ValueError, httpx.HTTPError) as exc:
        if created:
            destination.unlink(missing_ok=True)
        raise DwellError("setup_failed", f"Runtime archive download failed: {exc}") from exc

    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        if created:
            destination.unlink(missing_ok=True)
        raise DwellError(
            "setup_failed",
            "Runtime archive checksum verification failed.",
            details={"expected": expected_sha256, "actual": actual_sha256},
        )


def load_runtime_manifest() -> tuple[RuntimeSpec, ...]:
    resource = files("dwell.data").joinpath("runtimes.json")
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DwellError("setup_failed", f"Runtime manifest is unreadable: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise DwellError("setup_failed", "Runtime manifest has an unsupported schema.")
    try:
        runtimes = _RUNTIME_LIST.validate_python(payload.get("runtimes"))
    except Exception as exc:
        raise DwellError("setup_failed", f"Runtime manifest is invalid: {exc}") from exc
    if not runtimes:
        raise DwellError("setup_failed", "Runtime manifest contains no runtimes.")
    if len({runtime.id for runtime in runtimes}) != len(runtimes):
        raise DwellError("setup_failed", "Runtime manifest contains duplicate IDs.")
    return tuple(runtimes)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _normalized_remote(value: str) -> str:
    return value.strip().removesuffix("/")


def _first_line(value: str) -> str:
    lines = value.strip().splitlines()
    return lines[0] if lines else ""


def _git_environment(repository: Path | None = None) -> dict[str, str]:
    """Return a deterministic, non-interactive environment for local Git inspection."""

    environment = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    if repository is not None:
        git_directory = repository / ".git"
        environment.update(
            {
                # Use a packaged, inert Git directory so none of the checkout's
                # config, refs, hooks, or extensions are interpreted. Objects
                # and the index remain read-only inputs selected explicitly.
                "GIT_DIR": str(Path(__file__).resolve().parent / "data" / "safe_git"),
                "GIT_INDEX_FILE": str(git_directory / "index"),
                "GIT_OBJECT_DIRECTORY": str(git_directory / "objects"),
                "GIT_WORK_TREE": str(repository),
            }
        )
    return environment


def _read_small_regular_text(path: Path, *, limit: int) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise OSError("unsafe Git metadata file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(limit + 1)
        if len(payload) > limit:
            raise OSError("oversized Git metadata file")
    finally:
        os.close(descriptor)
    return payload.decode("utf-8")


def _read_runtime_identity(repository: Path) -> tuple[str | None, str | None]:
    git_directory = repository / ".git"
    if git_directory.is_symlink() or not git_directory.is_dir():
        return None, None
    try:
        head = _read_small_regular_text(git_directory / "HEAD", limit=256).strip()
        config_text = _read_small_regular_text(git_directory / "config", limit=64 * 1024)
        parser = configparser.RawConfigParser(interpolation=None, strict=True)
        parser.read_string(config_text)
        remote = parser.get('remote "origin"', "url", fallback=None)
    except (OSError, UnicodeError, configparser.Error):
        return None, None
    commit = head if _COMMIT_RE.fullmatch(head) else None
    return remote.strip() if remote and remote.strip() else None, commit


def _git_command(executable: str, repository: Path, *arguments: str) -> list[str]:
    return [
        executable,
        "--no-optional-locks",
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        f"core.hooksPath={os.devnull}",
        "-c",
        "core.untrackedCache=false",
        "-C",
        str(repository),
        *arguments,
    ]


def _parse_tree_entries(output: str) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for record in output.split("\0"):
        if not record:
            continue
        metadata, separator, name = record.partition("\t")
        fields = metadata.split()
        if separator != "\t" or len(fields) != 3:
            raise ValueError("invalid ls-tree record")
        mode, object_type, object_id = fields
        path = Path(name)
        if (
            object_type != "blob"
            or mode not in {"100644", "100755", "120000"}
            or not re.fullmatch(r"[0-9a-f]{40}", object_id)
            or path.is_absolute()
            or any(part in {"", ".", "..", ".git", ".venv"} for part in path.parts)
            or name in entries
        ):
            raise ValueError("unsafe ls-tree entry")
        entries[name] = (mode, object_id)
    return entries


def _parse_index_entries(output: str) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for record in output.split("\0"):
        if not record:
            continue
        metadata, separator, name = record.partition("\t")
        fields = metadata.split()
        if separator != "\t" or len(fields) != 3:
            raise ValueError("invalid ls-files record")
        mode, object_id, stage = fields
        if stage != "0" or name in entries:
            raise ValueError("unmerged or duplicate index entry")
        entries[name] = (mode, object_id)
    return entries


def _blob_object_id(path: Path, mode: str) -> tuple[str, bool]:
    metadata = path.lstat()
    if mode == "120000":
        if not stat.S_ISLNK(metadata.st_mode):
            raise OSError("tracked symlink has changed type")
        payload = os.fsencode(os.readlink(path))
        executable = False
    else:
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("tracked file has changed type")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise OSError("tracked file is no longer regular")
            digest = hashlib.sha1(  # noqa: S324 - Git SHA-1 object identity
                f"blob {opened.st_size}\0".encode(),
                usedforsecurity=False,
            )
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            payload_id = digest.hexdigest()
            executable = bool(opened.st_mode & 0o111)
        finally:
            os.close(descriptor)
        return payload_id, executable
    digest = hashlib.sha1(  # noqa: S324 - Git SHA-1 object identity
        f"blob {len(payload)}\0".encode(),
        usedforsecurity=False,
    )
    digest.update(payload)
    return digest.hexdigest(), executable


def _filesystem_entries(repository: Path) -> set[str]:
    entries: set[str] = set()
    pending = [repository]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as children:
            for child in children:
                path = Path(child.path)
                relative = path.relative_to(repository)
                if relative.parts[0] in {".git", ".venv"}:
                    continue
                if child.is_dir(follow_symlinks=False):
                    pending.append(path)
                else:
                    entries.add(relative.as_posix())
    return entries


def _runtime_worktree_is_dirty(
    repository: Path,
    *,
    expected_commit: str,
    executable: str,
    runner: Runner,
) -> bool:
    """Compare HEAD, index, and raw files without invoking repo filters or hooks."""

    environment = _git_environment(repository)
    integrity = runner(
        _git_command(
            executable,
            repository,
            "fsck",
            "--strict",
            "--no-references",
            "--no-reflogs",
            "--no-dangling",
            expected_commit,
        ),
        env=environment,
        timeout=120,
    )
    tree = runner(
        _git_command(
            executable,
            repository,
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            expected_commit,
        ),
        env=environment,
        timeout=30,
    )
    index = runner(
        _git_command(executable, repository, "ls-files", "--stage", "-z"),
        env=environment,
        timeout=30,
    )
    if integrity.returncode or tree.returncode or index.returncode:
        return True
    try:
        tree_entries = _parse_tree_entries(tree.stdout)
        index_entries = _parse_index_entries(index.stdout)
        if index_entries != tree_entries:
            return True
        if _filesystem_entries(repository) != set(tree_entries):
            return True
        for name, (mode, expected_object_id) in tree_entries.items():
            actual_object_id, executable_bit = _blob_object_id(repository / name, mode)
            if actual_object_id != expected_object_id:
                return True
            if (mode == "100755") != executable_bit:
                return True
    except (OSError, UnicodeError, ValueError):
        return True
    return False


def _sha256_regular_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("provenance input is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _runtime_source_payload(runtime: RuntimeSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": runtime.id,
        "commit": runtime.commit,
        "python": runtime.python,
    }
    if runtime.repository is not None:
        payload["repository"] = _normalized_remote(runtime.repository)
        payload["branch"] = runtime.branch
    if runtime.archive_url is not None:
        payload["archive_url"] = runtime.archive_url
        payload["archive_sha256"] = runtime.archive_sha256
    return payload


def _runtime_archive_source_payload(runtime: RuntimeSpec) -> dict[str, str]:
    if runtime.archive_url is None or runtime.archive_sha256 is None:
        raise ValueError("runtime does not define an archive")
    return {
        "id": runtime.id,
        "archive_url": runtime.archive_url,
        "archive_sha256": runtime.archive_sha256,
        "commit": runtime.commit,
        "python": runtime.python,
    }


def _archive_member_path(name: str, root: str | None) -> tuple[str, PurePosixPath | None]:
    raw = name.removesuffix("/")
    parts = raw.split("/")
    if not raw or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("archive contains an unsafe path")
    if root is not None and parts[0] != root:
        raise ValueError("archive contains multiple top-level directories")
    root = parts[0]
    if len(parts) == 1:
        return root, None
    relative = PurePosixPath(*parts[1:])
    if (
        relative.is_absolute()
        or len(relative.as_posix().encode()) > 1_024
        or relative.parts[0] in {".git", ".venv"}
        or relative.as_posix() in {_SOURCE_ARCHIVE_FILE, _SOURCE_MANIFEST_FILE}
    ):
        raise ValueError("archive contains a reserved or unsafe path")
    return root, relative


def _open_new_regular_file(path: Path, mode: int):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    return os.fdopen(descriptor, "wb")


def _archive_manifest_payload(
    archive: Path,
    *,
    source: Mapping[str, Any],
    extract_to: Path | None = None,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    root: str | None = None
    content_bytes = 0
    try:
        with tarfile.open(archive, mode="r:gz") as bundle:
            for index, member in enumerate(bundle):
                if index >= _MAX_ARCHIVE_ENTRIES:
                    raise ValueError("archive contains too many entries")
                root, relative = _archive_member_path(member.name, root)
                if relative is None:
                    if not member.isdir():
                        raise ValueError("archive root is not a directory")
                    continue
                name = relative.as_posix()
                if name in seen:
                    raise ValueError("archive contains duplicate paths")
                seen.add(name)
                destination = extract_to / Path(*relative.parts) if extract_to is not None else None
                if member.isdir():
                    if destination is not None:
                        destination.mkdir(parents=True, exist_ok=False)
                        destination.chmod(0o755)
                    continue
                if not member.isfile() or member.size < 0:
                    raise ValueError("archive contains links or unsupported file types")
                content_bytes += member.size
                if content_bytes > _MAX_ARCHIVE_CONTENT_BYTES:
                    raise ValueError("archive expands beyond the content limit")
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError("archive member could not be read")
                digest = hashlib.sha256()
                written = 0
                mode = 0o755 if member.mode & 0o111 else 0o644
                if destination is not None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    output = _open_new_regular_file(destination, mode)
                else:
                    output = None
                try:
                    while chunk := stream.read(1024 * 1024):
                        written += len(chunk)
                        if written > member.size:
                            raise ValueError("archive member exceeds its declared size")
                        digest.update(chunk)
                        if output is not None:
                            output.write(chunk)
                finally:
                    stream.close()
                    if output is not None:
                        output.close()
                if written != member.size:
                    raise ValueError("archive member is truncated")
                if destination is not None:
                    destination.chmod(mode)
                entries.append(
                    {
                        "path": name,
                        "sha256": digest.hexdigest(),
                        "mode": mode,
                        "size": member.size,
                    }
                )
    except (OSError, tarfile.TarError, ValueError) as exc:
        raise DwellError("setup_failed", f"Runtime archive is invalid: {exc}") from exc
    if root is None or not entries:
        raise DwellError("setup_failed", "Runtime archive contains no source files.")
    return {
        "schema_version": 1,
        "source": dict(source),
        "entries": sorted(entries, key=lambda item: item["path"]),
    }


def _read_source_manifest(runtime_dir: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(
            _read_small_regular_text(
                runtime_dir / _SOURCE_MANIFEST_FILE,
                limit=_SOURCE_MANIFEST_LIMIT,
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None
    source = payload.get("source")
    entries = payload.get("entries")
    if not isinstance(source, dict) or not isinstance(entries, list):
        return None
    if set(source) != _ARCHIVE_SOURCE_FIELDS:
        return None
    if not _valid_archive_source(source):
        return None
    return payload


def _valid_archive_source(source: Mapping[str, Any]) -> bool:
    if not all(
        isinstance(source.get(name), str) and source[name] for name in _ARCHIVE_SOURCE_FIELDS
    ):
        return False
    commit = source["commit"]
    archive_url = source["archive_url"]
    parsed = urllib.parse.urlsplit(archive_url)
    return bool(
        _COMMIT_RE.fullmatch(commit)
        and _SHA256_RE.fullmatch(source["archive_sha256"])
        and parsed.scheme == "https"
        and parsed.netloc
        and commit in archive_url
    )


def _read_recorded_archive_source(state_file: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(_read_small_regular_text(state_file, limit=_STRUCTURAL_FILE_LIMIT))
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return None
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        return None
    source = {name: runtime.get(name) for name in _ARCHIVE_SOURCE_FIELDS}
    return source if _valid_archive_source(source) else None


def _archive_filesystem_entries(runtime_dir: Path) -> set[str]:
    entries: set[str] = set()
    pending = [runtime_dir]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as children:
            for child in children:
                path = Path(child.path)
                relative = path.relative_to(runtime_dir)
                if relative.parts[0] == ".venv" or relative.as_posix() in {
                    _SOURCE_ARCHIVE_FILE,
                    _SOURCE_MANIFEST_FILE,
                }:
                    continue
                if child.is_dir(follow_symlinks=False):
                    pending.append(path)
                else:
                    entries.add(relative.as_posix())
    return entries


def _archive_source_is_dirty(
    runtime_dir: Path,
    *,
    trusted_source: Mapping[str, Any],
) -> bool:
    manifest = _read_source_manifest(runtime_dir)
    if manifest is None:
        return True
    source = manifest["source"]
    if source != dict(trusted_source):
        return True
    archive = runtime_dir / _SOURCE_ARCHIVE_FILE
    try:
        if _sha256_regular_file(archive) != source["archive_sha256"]:
            return True
        expected = _archive_manifest_payload(archive, source=source)
        if manifest != expected:
            return True
        expected_entries = {entry["path"]: entry for entry in expected["entries"]}
        if _archive_filesystem_entries(runtime_dir) != set(expected_entries):
            return True
        for name, entry in expected_entries.items():
            path = runtime_dir / Path(*PurePosixPath(name).parts)
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size != entry["size"]
                or (metadata.st_mode & 0o777) != entry["mode"]
                or _sha256_regular_file(path) != entry["sha256"]
            ):
                return True
    except (DwellError, OSError, TypeError, ValueError):
        return True
    return False


def _venv_tree_fingerprint(venv: Path, runtime: RuntimeSpec) -> str:
    allowed_symlinks = {
        Path("bin/python"),
        Path("bin/python3"),
        Path(f"bin/python{runtime.python}"),
    }
    records: list[bytes] = []
    pending = [venv]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as children:
            for child in children:
                path = Path(child.path)
                relative = path.relative_to(venv)
                if child.is_dir(follow_symlinks=False):
                    pending.append(path)
                    continue
                if child.name == _VENV_PROVENANCE_FILE:
                    continue
                metadata = child.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    if relative not in allowed_symlinks:
                        raise OSError(f"unexpected symlink in runtime venv: {relative}")
                    target = path.resolve(strict=True)
                    target_metadata = target.stat()
                    if not stat.S_ISREG(target_metadata.st_mode):
                        raise OSError(f"runtime Python symlink target is not a file: {relative}")
                    kind = b"symlink"
                    extra = b"\0".join(
                        (
                            os.fsencode(os.readlink(path)),
                            _sha256_regular_file(target).encode(),
                            str(target_metadata.st_mode & 0o777).encode(),
                        )
                    )
                elif stat.S_ISREG(metadata.st_mode):
                    kind = b"file"
                    extra = _sha256_regular_file(path).encode()
                else:
                    raise OSError("unsupported file type in runtime venv")
                records.append(
                    b"\0".join(
                        (
                            os.fsencode(relative.as_posix()),
                            kind,
                            str(metadata.st_mode & 0o777).encode(),
                            str(metadata.st_size).encode(),
                            str(metadata.st_mtime_ns).encode(),
                            extra,
                        )
                    )
                )
    digest = hashlib.sha256()
    for record in sorted(records):
        digest.update(record)
        digest.update(b"\n")
    return digest.hexdigest()


def _venv_provenance_payload(runtime_dir: Path, runtime: RuntimeSpec) -> dict[str, Any]:
    venv = runtime_dir / ".venv"
    python = venv / "bin" / "python"
    cli = venv / "bin" / "ltx-2-mlx"
    lockfile = runtime_dir / "uv.lock"
    python_for_hash = python.resolve(strict=True) if python.is_symlink() else python
    return {
        "schema_version": 1,
        "runtime": _runtime_source_payload(runtime),
        "runtime_dir": str(runtime_dir),
        "venv_dir": str(venv),
        "uv_lock_sha256": _sha256_regular_file(lockfile),
        "python_sha256": _sha256_regular_file(python_for_hash),
        "cli_sha256": _sha256_regular_file(cli),
        "tree_fingerprint": _venv_tree_fingerprint(venv, runtime),
    }


def _venv_provenance_is_valid(runtime_dir: Path, runtime: RuntimeSpec) -> bool:
    marker = runtime_dir / ".venv" / _VENV_PROVENANCE_FILE
    try:
        actual = json.loads(_read_small_regular_text(marker, limit=_STRUCTURAL_FILE_LIMIT))
        expected = _venv_provenance_payload(runtime_dir, runtime)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(actual, dict) and actual == expected


class SetupManager:
    def __init__(
        self,
        config: DwellConfig | None = None,
        *,
        runtime: RuntimeSpec | None = None,
        runner: Runner = _run,
        doctor_runner: DoctorRunner = run_doctor,
        archive_downloader: ArchiveDownloader = _download_archive,
    ) -> None:
        self.config = config or DwellConfig.from_env()
        self.runtime = runtime or self._default_runtime()
        self.runner = runner
        self.doctor_runner = doctor_runner
        self.archive_downloader = archive_downloader
        self._tools: dict[str, str] = {}

    @staticmethod
    def _default_runtime() -> RuntimeSpec:
        runtimes = load_runtime_manifest()
        matching = [runtime for runtime in runtimes if runtime.id == "ltx-2-mlx"]
        if len(matching) != 1:
            raise DwellError("setup_failed", "Manifest must define exactly one ltx-2-mlx runtime.")
        return matching[0]

    @property
    def runtime_dir(self) -> Path:
        return self.config.runtimes_dir / self.runtime.id

    def run(self, mode: SetupMode = SetupMode.INSTALL, *, force: bool = False) -> SetupReport:
        if force and mode != SetupMode.UPGRADE:
            raise DwellError("invalid_request", "force is only valid for runtime upgrades.")
        messages = self._preflight()
        if mode == SetupMode.CHECK:
            # A read-only check must never import or execute code from the
            # runtime checkout or its generated virtual environment.
            inspection = self.inspect(probe_venv=False)
            messages.extend(self._inspection_messages(inspection))
            state_current = self._state_matches()
            messages.append(
                "Setup state: current"
                if state_current
                else f"Setup state: missing or stale ({self.config.setup_state_file})"
            )
            return SetupReport(
                mode=mode,
                changed=False,
                healthy=(
                    self._matches_manifest(inspection) and inspection.venv_healthy and state_current
                ),
                messages=tuple(messages),
            )

        # The lock itself lives in state/, so create only that parent before
        # taking the lease. The remainder of the layout is protected by it.
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        with self._setup_lock():
            self.config.ensure_layout()
            inspection = self.inspect(probe_venv=False)
            if self._runtime_mutation_required(mode, inspection, force=force):
                # Match the lock order used by model removal (lifecycle, then
                # LTX) and retain both leases for every runtime mutation.
                with self._runtime_mutation_locks():
                    self._refuse_active_work()
                    # Re-inspect under the leases; the checkout may have changed
                    # while setup was waiting to acquire them.
                    outcome = self._mutate(mode, force=force)
                    checks, state_changed, completion_messages = self._complete_setup(outcome)
            else:
                # Healthy no-ops and validation errors do not need to disrupt a
                # running server. Never permit this unlocked path to mutate.
                outcome = self._mutate(mode, inspection=inspection, force=force)
                current = self.inspect(probe_venv=False)
                if outcome.changed or current != inspection:
                    raise DwellError(
                        "setup_failed",
                        "Runtime changed during setup inspection; retry the command.",
                    )
                checks, state_changed, completion_messages = self._complete_setup(outcome)
            messages.extend(outcome.messages)
            messages.extend(completion_messages)
            return SetupReport(
                mode=mode,
                changed=outcome.changed or state_changed,
                healthy=True,
                messages=tuple(messages),
                doctor_checks=tuple(checks),
            )

    def _complete_setup(
        self,
        outcome: MutationOutcome,
    ) -> tuple[list[DoctorCheck], bool, list[str]]:
        try:
            checks = self.doctor_runner(self.config)
            if has_errors(checks):
                failures = [check.detail for check in checks if check.level.value == "error"]
                raise DwellError(
                    "setup_failed",
                    "Setup completed its file operations, but dwell doctor reported errors.",
                    details={"errors": failures},
                )
            if (
                outcome.backup is not None
                and outcome.previous_inspection is not None
                and not outcome.preserve_backup
            ):
                self._require_runtime_unchanged(outcome.previous_inspection, outcome.backup)
            state_changed = not self._state_matches()
            if state_changed:
                self._write_state()
                messages = [f"Setup state written to {self.config.setup_state_file}"]
            else:
                messages = ["Setup state is already current"]
        except BaseException as exc:
            if outcome.backup is not None:
                self._rollback_completed_upgrade(outcome, exc)
                raise DwellError(
                    "setup_failed",
                    "Runtime upgrade verification failed; the previous runtime was restored.",
                    details={"error": str(exc)},
                ) from exc
            raise

        if outcome.backup is not None and outcome.preserve_backup:
            messages.append(f"Previous runtime preserved at {outcome.backup}")
        elif outcome.backup is not None:
            try:
                shutil.rmtree(outcome.backup)
            except OSError as exc:
                messages.append(
                    f"Runtime is healthy, but backup cleanup failed at {outcome.backup}: {exc}"
                )
        return checks, state_changed, messages

    def _rollback_completed_upgrade(self, outcome: MutationOutcome, cause: BaseException) -> None:
        backup = outcome.backup
        if backup is None:
            return
        failed = self.config.runtimes_dir / f".{self.runtime.id}.failed-{os.getpid()}"
        if failed.exists() or failed.is_symlink():
            raise DwellError(
                "setup_failed",
                "Runtime verification failed and rollback was blocked by an existing path.",
                details={"error": str(cause), "backup": str(backup), "failed": str(failed)},
            ) from cause
        moved_new = False
        try:
            if self.runtime_dir.exists() or self.runtime_dir.is_symlink():
                os.replace(self.runtime_dir, failed)
                moved_new = True
            os.replace(backup, self.runtime_dir)
        except BaseException as rollback_exc:
            raise DwellError(
                "setup_failed",
                "Runtime verification failed and rollback also failed; both paths were preserved.",
                details={
                    "verification_error": str(cause),
                    "rollback_error": str(rollback_exc),
                    "backup": str(backup),
                    "failed": str(failed) if moved_new else None,
                },
            ) from cause
        if moved_new:
            try:
                self._cleanup_failed_runtime(failed)
            except BaseException as cleanup_exc:
                raise DwellError(
                    "setup_failed",
                    "The previous runtime was restored, but failed-runtime cleanup "
                    "did not complete.",
                    details={
                        "verification_error": str(cause),
                        "cleanup_error": str(cleanup_exc),
                        "failed": str(failed),
                    },
                ) from cause

    def _cleanup_failed_runtime(self, failed: Path) -> None:
        if failed.parent != self.config.runtimes_dir or not failed.name.startswith(
            f".{self.runtime.id}.failed-"
        ):
            raise DwellError("setup_failed", f"Refusing unsafe failed-runtime cleanup: {failed}")
        if failed.is_symlink():
            failed.unlink()
        elif failed.exists():
            shutil.rmtree(failed)

    def _preflight(self) -> list[str]:
        system = platform.system()
        machine = platform.machine()
        if system != "Darwin" or machine != "arm64":
            raise DwellError(
                "unsupported_platform",
                f"Dwell setup requires macOS on Apple Silicon; detected {system} {machine}.",
            )
        macos_version = platform.mac_ver()[0]
        try:
            macos_major = int(macos_version.split(".", 1)[0])
        except (TypeError, ValueError):
            macos_major = 0
        if macos_major < 14:
            raise DwellError(
                "unsupported_platform",
                "Dwell setup requires macOS 14 Sonoma or newer on Apple Silicon; "
                f"detected macOS {macos_version or 'unknown'}.",
            )
        unavailable: dict[str, str] = {}
        required_tools = dict(_TOOL_VERSION_ARGUMENTS)
        if not self.runtime.uses_archive:
            required_tools["git"] = ("--version",)
        for name, version_arguments in required_tools.items():
            executable = shutil.which(name)
            if executable is None:
                unavailable[name] = "not found"
                continue
            result = self.runner([executable, *version_arguments], timeout=15)
            if result.returncode:
                unavailable[name] = (
                    _first_line(result.stderr)
                    or _first_line(result.stdout)
                    or f"exit {result.returncode}"
                )
                continue
            self._tools[name] = executable
        if unavailable:
            names = ", ".join(unavailable)
            raise DwellError(
                "missing_dependency",
                f"Required setup tools are missing or unusable: {names}.",
                details={"unavailable": unavailable},
            )

        tool_names = ", ".join(required_tools)
        messages = ["Platform: macOS 14+ arm64", f"Tools: {tool_names}"]
        memory = self._physical_memory_gib()
        if memory is None:
            messages.append("Memory: unable to determine unified-memory capacity")
        elif memory < 48:
            messages.append(f"Memory warning: {memory:.1f} GiB detected; bf16 needs about 48+ GB")
        else:
            messages.append(f"Memory: {memory:.1f} GiB detected")
        free = self._free_disk_gib()
        if free is None:
            messages.append("Disk: unable to determine free capacity")
        elif free < 80:
            messages.append(f"Disk warning: {free:.1f} GiB free; allow room for runtime and models")
        else:
            messages.append(f"Disk: {free:.1f} GiB free")
        return messages

    def _physical_memory_gib(self) -> float | None:
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, TypeError, ValueError):
            return None
        if pages <= 0 or page_size <= 0:
            return None
        return pages * page_size / (1024**3)

    def _free_disk_gib(self) -> float | None:
        candidate = self.config.home
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        try:
            return shutil.disk_usage(candidate).free / (1024**3)
        except OSError:
            return None

    @contextmanager
    def _setup_lock(self) -> Iterator[None]:
        with self.config.setup_lock_file.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DwellError(
                    "setup_busy",
                    "Another dwell setup operation is already in progress.",
                    status_code=409,
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def _runtime_mutation_locks(self) -> Iterator[None]:
        with self._exclusive_runtime_lock(
            self.config.start_lock_file,
            "server_busy",
            "A Dwell service lifecycle operation is in progress; setup was refused.",
        ):
            with self._exclusive_runtime_lock(
                self.config.ltx_lock_file,
                "server_busy",
                "An LTX job or runtime operation is in progress; setup was refused.",
            ):
                yield

    @staticmethod
    @contextmanager
    def _exclusive_runtime_lock(path: Path, code: str, message: str) -> Iterator[None]:
        with path.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise DwellError(code, message, status_code=409) from exc
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def inspect(self, *, probe_venv: bool = False) -> RuntimeInspection:
        return self._inspect_runtime(self.runtime_dir, probe_venv=probe_venv)

    def _inspect_runtime(
        self,
        runtime_dir: Path,
        *,
        probe_venv: bool,
    ) -> RuntimeInspection:
        if not runtime_dir.exists():
            return RuntimeInspection(
                exists=False,
                source_valid=False,
                remote=None,
                commit=None,
                source_hash=None,
                dirty=False,
                venv_exists=False,
                venv_healthy=False,
                problems=("missing",),
            )
        if runtime_dir.is_symlink():
            return RuntimeInspection(
                exists=True,
                source_valid=False,
                remote=None,
                commit=None,
                source_hash=None,
                dirty=True,
                venv_exists=False,
                venv_healthy=False,
                problems=("runtime path is a symlink",),
            )
        if not runtime_dir.is_dir():
            return RuntimeInspection(
                exists=True,
                source_valid=False,
                remote=None,
                commit=None,
                source_hash=None,
                dirty=True,
                venv_exists=False,
                venv_healthy=False,
                problems=("runtime path is not a directory",),
            )

        problems: list[str] = []
        source_hash: str | None = None
        git_directory = runtime_dir / ".git"
        if self.runtime.uses_archive and not git_directory.exists():
            manifest = _read_source_manifest(runtime_dir)
            if manifest is None:
                remote = None
                commit = None
                dirty = True
                source_valid = False
                problems.append("runtime archive source manifest is missing or unreadable")
            else:
                source = manifest["source"]
                remote = source["archive_url"]
                commit = source["commit"]
                source_hash = source["archive_sha256"]
                expected_source = _runtime_archive_source_payload(self.runtime)
                recorded_source = _read_recorded_archive_source(self.config.setup_state_file)
                trusted_source = (
                    expected_source
                    if source == expected_source
                    else recorded_source
                    if recorded_source is not None and source == recorded_source
                    else None
                )
                dirty = trusted_source is None or _archive_source_is_dirty(
                    runtime_dir,
                    trusted_source=trusted_source or expected_source,
                )
                if source != expected_source:
                    problems.append("runtime archive identity does not match the manifest")
                if dirty:
                    problems.append("runtime archive source has local or unsafe changes")
                source_valid = source == expected_source and not dirty
        else:
            if not git_directory.exists():
                return RuntimeInspection(
                    exists=True,
                    source_valid=False,
                    remote=None,
                    commit=None,
                    source_hash=None,
                    dirty=True,
                    venv_exists=False,
                    venv_healthy=False,
                    problems=("runtime path is not a recognized source checkout",),
                )
            # Read canonical clone metadata as bounded plain data. No Git command
            # sees this checkout's config during inspection.
            remote, commit = _read_runtime_identity(runtime_dir)
            expected_repository = self.runtime.repository
            remote_ok = (
                remote is not None
                and expected_repository is not None
                and _normalized_remote(remote) == _normalized_remote(expected_repository)
            )
            commit_ok = commit == self.runtime.commit
            if remote is None:
                problems.append("origin remote is unreadable")
            elif not remote_ok:
                problems.append(f"origin remote is {remote}; expected {expected_repository}")
            if commit is None or not _COMMIT_RE.fullmatch(commit):
                problems.append("HEAD commit is unreadable")
            elif not commit_ok:
                problems.append(f"HEAD is {commit}; expected {self.runtime.commit}")
            worktree_dirty = commit is None or _runtime_worktree_is_dirty(
                runtime_dir,
                expected_commit=commit or self.runtime.commit,
                executable=self._tools.get("git") or shutil.which("git") or "git",
                runner=self.runner,
            )
            migrating_legacy_source = self.runtime.uses_archive
            # Archive migration is allowed only for the exact trusted legacy
            # checkout. A mismatched remote/commit is not treated as replaceable.
            dirty = worktree_dirty or (migrating_legacy_source and not (remote_ok and commit_ok))
            if worktree_dirty:
                problems.append("runtime checkout has local changes")
            if migrating_legacy_source:
                problems.append("runtime uses legacy Git source; run dwell setup --upgrade")
                source_valid = False
            else:
                source_valid = remote_ok and commit_ok and not dirty

        venv = runtime_dir / ".venv"
        # Path.exists() is false for a dangling symlink, but it remains an
        # unsafe path that setup must not hand to uv or delete implicitly.
        venv_exists = venv.exists() or venv.is_symlink()
        if venv.is_symlink():
            problems.append("runtime virtual environment is a symlink")
            venv_healthy = False
        else:
            venv_healthy = (
                source_valid
                and self._venv_is_structurally_healthy(runtime_dir)
                and _venv_provenance_is_valid(runtime_dir, self.runtime)
            )
            if venv_healthy and probe_venv and source_valid:
                venv_healthy = self._venv_probe_passes(runtime_dir)
            if not venv_healthy:
                problems.append("runtime virtual environment is missing or unhealthy")
        return RuntimeInspection(
            exists=True,
            source_valid=source_valid,
            remote=remote,
            commit=commit,
            source_hash=source_hash,
            dirty=dirty,
            venv_exists=venv_exists,
            venv_healthy=venv_healthy,
            problems=tuple(problems),
        )

    def _venv_is_structurally_healthy(self, runtime_dir: Path) -> bool:
        venv = runtime_dir / ".venv"
        python = runtime_dir / ".venv" / "bin" / "python"
        cli = runtime_dir / ".venv" / "bin" / "ltx-2-mlx"
        metadata = venv / "pyvenv.cfg"
        if venv.is_symlink() or not venv.is_dir():
            return False
        if not python.is_file() or not os.access(python, os.X_OK):
            return False
        if cli.is_symlink() or not cli.is_file() or not os.access(cli, os.X_OK):
            return False
        try:
            if metadata.is_symlink() or metadata.stat().st_size > _STRUCTURAL_FILE_LIMIT:
                return False
            metadata_text = metadata.read_text(encoding="utf-8")
            if cli.stat().st_size > _STRUCTURAL_FILE_LIMIT:
                return False
            cli_text = cli.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return False
        versions = {
            value.strip()
            for line in metadata_text.splitlines()
            if "=" in line
            for key, value in (line.split("=", 1),)
            if key.strip() in {"version", "version_info"}
        }
        expected_python = f"{self.runtime.python}."
        if not any(version.startswith(expected_python) for version in versions):
            return False
        lines = cli_text.splitlines()
        # distlib/uv uses a /bin/sh launcher when the absolute interpreter path
        # needs shell quoting (for example when DWELL_HOME contains spaces).
        header = "\n".join(lines[:5])
        if str(python) not in header:
            return False
        if "from ltx_pipelines_mlx.cli import main" not in cli_text:
            return False
        return True

    def _venv_probe_passes(self, runtime_dir: Path) -> bool:
        python = runtime_dir / ".venv" / "bin" / "python"
        cli = runtime_dir / ".venv" / "bin" / "ltx-2-mlx"
        python_result = self.runner([str(python), "--version"], cwd=runtime_dir, timeout=15)
        version = python_result.stdout or python_result.stderr
        if python_result.returncode or not version.startswith(f"Python {self.runtime.python}"):
            return False
        environment = self.config.subprocess_env()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        cli_result = self.runner(
            [str(cli), "generate", "--help"],
            cwd=runtime_dir,
            env=environment,
            timeout=30,
        )
        return cli_result.returncode == 0

    def _matches_manifest(self, inspection: RuntimeInspection) -> bool:
        return inspection.exists and inspection.source_valid and not inspection.dirty

    def _inspection_messages(self, inspection: RuntimeInspection) -> list[str]:
        if not inspection.exists:
            return [f"Runtime: missing ({self.runtime_dir})"]
        if not inspection.problems:
            return [f"Runtime: healthy at {self.runtime.commit}"]
        return [f"Runtime problem: {problem}" for problem in inspection.problems]

    def _runtime_mutation_required(
        self,
        mode: SetupMode,
        inspection: RuntimeInspection,
        *,
        force: bool = False,
    ) -> bool:
        if inspection.exists and inspection.dirty:
            return mode == SetupMode.UPGRADE and force and not self.runtime_dir.is_symlink()
        venv_is_symlink = (self.runtime_dir / ".venv").is_symlink()
        if venv_is_symlink:
            return False
        if mode == SetupMode.REPAIR:
            return (
                inspection.exists
                and inspection.source_valid
                and not inspection.venv_healthy
                and not venv_is_symlink
            )
        if mode == SetupMode.UPGRADE:
            return (
                not inspection.exists
                or not inspection.source_valid
                or (not inspection.venv_healthy and inspection.source_valid and not venv_is_symlink)
            )
        return not inspection.exists or (
            inspection.source_valid
            and not inspection.venv_healthy
            and (not inspection.venv_exists or not self.config.setup_state_file.exists())
        )

    def _mutate(
        self,
        mode: SetupMode,
        *,
        inspection: RuntimeInspection | None = None,
        force: bool = False,
    ) -> MutationOutcome:
        inspection = inspection or self.inspect(probe_venv=False)
        if self.runtime_dir.is_symlink():
            raise DwellError(
                "runtime_unhealthy",
                "Refusing to replace a symlinked runtime path.",
            )
        if inspection.exists and inspection.dirty and not (mode == SetupMode.UPGRADE and force):
            raise DwellError(
                "dirty_runtime",
                "The LTX runtime has local or unsafe changes; setup refused to overwrite it. "
                "Use dwell setup --upgrade --force to preserve and replace it.",
                details={"path": str(self.runtime_dir), "problems": inspection.problems},
            )
        if (self.runtime_dir / ".venv").is_symlink():
            raise DwellError(
                "runtime_unhealthy",
                "Refusing to use a symlinked runtime virtual environment.",
            )
        if mode == SetupMode.REPAIR:
            return self._repair(inspection)
        if mode == SetupMode.UPGRADE:
            return self._upgrade(inspection, force=force)
        return self._install_missing(inspection)

    def _install_missing(self, inspection: RuntimeInspection) -> MutationOutcome:
        if not inspection.exists:
            self._install_source()
            self._sync_runtime()
            self._require_healthy_runtime()
            return MutationOutcome(True, (f"Installed {self.runtime.id} at {self.runtime.commit}",))
        if not inspection.source_valid:
            raise DwellError(
                "runtime_mismatch",
                "Installed runtime does not match the manifest; use dwell setup --upgrade.",
                details={"problems": inspection.problems},
            )
        if inspection.venv_healthy:
            return MutationOutcome(
                False,
                (f"Runtime already healthy at {self.runtime.commit}; nothing to do",),
            )
        if (self.runtime_dir / ".venv").is_symlink():
            raise DwellError(
                "runtime_unhealthy",
                "Refusing to use a symlinked runtime virtual environment.",
            )
        if inspection.venv_exists:
            if not self.config.setup_state_file.exists():
                venv = self.runtime_dir / ".venv"
                if venv.is_symlink():
                    raise DwellError(
                        "runtime_unhealthy",
                        "Refusing to remove a symlinked runtime virtual environment.",
                    )
                shutil.rmtree(venv)
                self._sync_runtime()
                self._require_healthy_runtime()
                return MutationOutcome(
                    True,
                    ("Rebuilt the incomplete runtime virtual environment",),
                )
            raise DwellError(
                "runtime_unhealthy",
                "Runtime venv is unhealthy; run dwell setup --repair.",
            )
        self._sync_runtime()
        self._require_healthy_runtime()
        return MutationOutcome(True, ("Completed the missing runtime virtual environment",))

    def _repair(self, inspection: RuntimeInspection) -> MutationOutcome:
        if not inspection.exists:
            raise DwellError("runtime_missing", "Runtime is missing; run dwell setup first.")
        if not inspection.source_valid:
            raise DwellError(
                "runtime_mismatch",
                "Repair requires the pinned manifest source; use dwell setup --upgrade.",
                details={"problems": inspection.problems},
            )
        if inspection.venv_healthy:
            return MutationOutcome(False, ("Runtime venv is already healthy; nothing to repair",))
        venv = self.runtime_dir / ".venv"
        if venv.is_symlink():
            raise DwellError("runtime_unhealthy", "Refusing to remove a symlinked runtime venv.")
        if venv.exists():
            shutil.rmtree(venv)
        self._sync_runtime()
        self._require_healthy_runtime()
        return MutationOutcome(True, ("Rebuilt and verified the runtime virtual environment",))

    def _upgrade(
        self,
        inspection: RuntimeInspection,
        *,
        force: bool = False,
    ) -> MutationOutcome:
        if not inspection.exists:
            self._install_source()
            self._sync_runtime()
            self._require_healthy_runtime()
            return MutationOutcome(True, (f"Installed {self.runtime.id} at {self.runtime.commit}",))
        if inspection.source_valid and inspection.venv_healthy:
            return MutationOutcome(
                False,
                (f"Runtime already matches {self.runtime.commit}; nothing to upgrade",),
            )
        if inspection.source_valid:
            return self._repair(inspection)

        staging = self._clone_staging()
        preserving_dirty_runtime = force and inspection.dirty
        if preserving_dirty_runtime:
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            backup = self.config.runtimes_dir / (f".{self.runtime.id}.forced-backup-{timestamp}")
        else:
            backup = self.config.runtimes_dir / f".{self.runtime.id}.backup-{os.getpid()}"
        if backup.exists() or backup.is_symlink():
            self._cleanup_staging(staging)
            raise DwellError(
                "setup_failed", f"Refusing to overwrite existing backup path: {backup}"
            )
        moved_old = False
        moved_new = False
        try:
            # Staging can take a long time. Recheck both active state and the
            # exact checkout identity immediately before replacing anything.
            self._refuse_active_work()
            if not preserving_dirty_runtime:
                self._require_runtime_unchanged(inspection, self.runtime_dir)
            os.replace(self.runtime_dir, backup)
            moved_old = True
            os.replace(staging, self.runtime_dir)
            moved_new = True
            self._sync_runtime()
            self._require_healthy_runtime()
            # Preserve edits made through an already-open handle after the
            # checkout was renamed to its backup path.
            if not preserving_dirty_runtime:
                self._require_runtime_unchanged(inspection, backup)
        except BaseException as exc:
            rollback_error: Exception | None = None
            try:
                if moved_new and self.runtime_dir.exists():
                    shutil.rmtree(self.runtime_dir)
                if moved_old and backup.exists():
                    os.replace(backup, self.runtime_dir)
            except BaseException as rollback_exc:  # pragma: no cover - catastrophic failure
                rollback_error = rollback_exc
            if rollback_error is not None:
                raise DwellError(
                    "setup_failed",
                    "Runtime upgrade failed and rollback also failed.",
                    details={"upgrade_error": str(exc), "rollback_error": str(rollback_error)},
                ) from exc
            if isinstance(exc, DwellError) and not moved_old:
                raise
            if isinstance(exc, DwellError) and exc.code == "dirty_runtime":
                raise DwellError(
                    "dirty_runtime",
                    "Runtime changed during upgrade; the previous runtime was preserved.",
                    details=exc.details,
                ) from exc
            raise DwellError(
                "setup_failed",
                "Runtime upgrade failed; the previous runtime was restored.",
                details={"error": str(exc)},
            ) from exc
        finally:
            if staging.exists():
                self._cleanup_staging(staging)
        return MutationOutcome(
            True,
            (f"Upgraded {self.runtime.id} to {self.runtime.commit}",),
            backup=backup,
            previous_inspection=None if preserving_dirty_runtime else inspection,
            preserve_backup=preserving_dirty_runtime,
        )

    def _refuse_active_work(self) -> None:
        from dwell.process import (
            _read_ltx_child_state,
            _recorded_ltx_group_may_be_active,
            read_server_state,
        )

        server = read_server_state(self.config, clean_stale=False)
        if server is not None:
            raise DwellError(
                "server_busy",
                "Stop Dwell before upgrading its runtime.",
                details={"pid": server.pid},
                status_code=409,
            )
        child = _read_ltx_child_state(self.config)
        if child is not None and _recorded_ltx_group_may_be_active(child):
            raise DwellError(
                "server_busy",
                "An LTX job may still be active; runtime upgrade was refused.",
                details={"pid": child["pid"], "job_id": child["job_id"]},
                status_code=409,
            )

    def _require_runtime_unchanged(
        self,
        expected: RuntimeInspection,
        path: Path,
    ) -> None:
        current = self._inspect_runtime(path, probe_venv=False)
        changed = (
            not current.exists
            or current.dirty
            or current.remote != expected.remote
            or current.commit != expected.commit
            or current.source_hash != expected.source_hash
        )
        if changed:
            raise DwellError(
                "dirty_runtime",
                "The installed runtime changed while upgrade was in progress.",
                details={
                    "path": str(path),
                    "expected_remote": expected.remote,
                    "actual_remote": current.remote,
                    "expected_commit": expected.commit,
                    "actual_commit": current.commit,
                    "expected_source_hash": expected.source_hash,
                    "actual_source_hash": current.source_hash,
                    "problems": current.problems,
                },
            )

    def _install_source(self) -> None:
        staging = self._clone_staging()
        try:
            if self.runtime_dir.exists() or self.runtime_dir.is_symlink():
                raise DwellError(
                    "runtime_mismatch",
                    f"Refusing to overwrite existing runtime path: {self.runtime_dir}",
                )
            os.replace(staging, self.runtime_dir)
        finally:
            if staging.exists():
                self._cleanup_staging(staging)

    def _clone_staging(self) -> Path:
        staging = Path(tempfile.mkdtemp(prefix=f"{self.runtime.id}-", dir=self.config.tmp_dir))
        # Prepare source only. The venv is deliberately created after the move
        # to its final path because generated console scripts contain absolute paths.
        if self.runtime.uses_archive:
            assert self.runtime.archive_url is not None
            assert self.runtime.archive_sha256 is not None
            archive = staging / _SOURCE_ARCHIVE_FILE
            try:
                self.archive_downloader(
                    self.runtime.archive_url,
                    archive,
                    self.runtime.archive_sha256,
                )
                if _sha256_regular_file(archive) != self.runtime.archive_sha256:
                    raise DwellError(
                        "setup_failed", "Runtime archive checksum verification failed."
                    )
                manifest = _archive_manifest_payload(
                    archive,
                    source=_runtime_archive_source_payload(self.runtime),
                    extract_to=staging,
                )
                _atomic_write_json(staging / _SOURCE_MANIFEST_FILE, manifest)
                self._verify_staging(staging)
            except BaseException:
                self._cleanup_staging(staging)
                raise
            return staging

        clone = self.runner(
            [
                self._tools["git"],
                "clone",
                "--no-checkout",
                "--single-branch",
                "--branch",
                self.runtime.branch,
                self.runtime.repository,
                str(staging),
            ],
            env=_git_environment(),
            timeout=1_800,
        )
        if clone.returncode:
            self._cleanup_staging(staging)
            raise self._command_error("Runtime clone failed", clone)
        checkout = self._git(staging, "checkout", "--detach", self.runtime.commit, timeout=120)
        if checkout.returncode:
            self._cleanup_staging(staging)
            raise self._command_error("Pinned runtime checkout failed", checkout)
        self._verify_staging(staging)
        return staging

    def _verify_staging(self, staging: Path) -> None:
        if self.runtime.uses_archive:
            manifest = _read_source_manifest(staging)
            if manifest is None or manifest["source"] != _runtime_archive_source_payload(
                self.runtime
            ):
                self._cleanup_staging(staging)
                raise DwellError("runtime_mismatch", "Runtime archive identity is invalid.")
            if _archive_source_is_dirty(
                staging,
                trusted_source=_runtime_archive_source_payload(self.runtime),
            ):
                self._cleanup_staging(staging)
                raise DwellError("dirty_runtime", "Extracted runtime archive failed verification.")
            return

        remote, commit = _read_runtime_identity(staging)
        dirty = _runtime_worktree_is_dirty(
            staging,
            expected_commit=commit or self.runtime.commit,
            executable=self._tools.get("git") or shutil.which("git") or "git",
            runner=self.runner,
        )
        expected_repository = self.runtime.repository
        if (
            remote is None
            or expected_repository is None
            or _normalized_remote(remote) != _normalized_remote(expected_repository)
        ):
            self._cleanup_staging(staging)
            raise DwellError("runtime_mismatch", "Cloned runtime remote verification failed.")
        if commit != self.runtime.commit:
            self._cleanup_staging(staging)
            raise DwellError("runtime_mismatch", "Cloned runtime commit verification failed.")
        if dirty:
            self._cleanup_staging(staging)
            raise DwellError("dirty_runtime", "Newly cloned runtime is unexpectedly dirty.")

    def _sync_runtime(self) -> None:
        environment = dict(os.environ)
        for name in tuple(environment):
            if name.startswith(("UV_", "PIP_")) or name == "VIRTUAL_ENV":
                environment.pop(name)
        environment.update(
            {
                "DWELL_HOME": str(self.config.home),
                "HF_HOME": str(self.config.hf_home),
                "HF_HUB_CACHE": str(self.config.hf_hub_cache),
                "HF_XET_CACHE": str(self.config.hf_xet_cache),
                "UV_PROJECT": str(self.runtime_dir),
                "UV_PROJECT_ENVIRONMENT": str(self.runtime_dir / ".venv"),
                "UV_PYTHON_DOWNLOADS": "automatic",
            }
        )
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            environment.pop(name, None)
        result = self.runner(
            [
                self._tools["uv"],
                "sync",
                "--project",
                str(self.runtime_dir),
                "--locked",
                # Probing an editable install imports from the integrity-checked
                # source tree and leaves __pycache__ files behind, making a fresh
                # runtime immediately appear modified.
                "--no-editable",
                "--python",
                self.runtime.python,
            ],
            cwd=self.runtime_dir,
            env=environment,
            timeout=1_800,
        )
        if result.returncode:
            raise self._command_error("uv sync --locked failed", result)

    def _require_healthy_runtime(self) -> None:
        # Every caller invokes this immediately after a pinned `uv sync`. Only
        # then is it safe and useful to execute the generated console script.
        inspection = self.inspect(probe_venv=False)
        probe_passed = (
            inspection.source_valid
            and self._venv_is_structurally_healthy(self.runtime_dir)
            and self._venv_probe_passes(self.runtime_dir)
        )
        if not probe_passed:
            raise DwellError(
                "runtime_unhealthy",
                "Runtime verification failed after setup.",
                details={"problems": inspection.problems},
            )
        _atomic_write_json(
            self.runtime_dir / ".venv" / _VENV_PROVENANCE_FILE,
            _venv_provenance_payload(self.runtime_dir, self.runtime),
        )
        if not self.inspect(probe_venv=False).venv_healthy:
            raise DwellError(
                "runtime_unhealthy",
                "Runtime provenance verification failed after setup.",
            )

    def _git(
        self,
        repository: Path,
        *arguments: str,
        timeout: float = 30,
    ) -> subprocess.CompletedProcess[str]:
        git = self._tools.get("git") or shutil.which("git") or "git"
        return self.runner(
            _git_command(git, repository, *arguments),
            env=_git_environment(),
            timeout=timeout,
        )

    @staticmethod
    def _command_error(message: str, result: subprocess.CompletedProcess[str]) -> DwellError:
        detail = (
            _first_line(result.stderr) or _first_line(result.stdout) or f"exit {result.returncode}"
        )
        return DwellError("setup_failed", f"{message}: {detail}")

    def _cleanup_staging(self, staging: Path) -> None:
        if staging.parent != self.config.tmp_dir or not staging.name.startswith(
            f"{self.runtime.id}-"
        ):
            raise DwellError("setup_failed", f"Refusing unsafe staging cleanup: {staging}")
        if staging.is_symlink():
            staging.unlink()
        elif staging.exists():
            shutil.rmtree(staging)

    def _write_state(self) -> None:
        _atomic_write_json(
            self.config.setup_state_file,
            {
                "schema_version": 1,
                "dwell_version": __version__,
                "runtime": _runtime_source_payload(self.runtime),
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )

    def _state_matches(self) -> bool:
        try:
            payload = json.loads(self.config.setup_state_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        runtime = payload.get("runtime")
        return (
            payload.get("schema_version") == 1
            and payload.get("dwell_version") == __version__
            and isinstance(runtime, dict)
            and runtime == _runtime_source_payload(self.runtime)
        )
