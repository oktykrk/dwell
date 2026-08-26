from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from dwell.config import DwellConfig
from dwell.errors import DwellError
from dwell.process import server_status, stop_server

_REPOSITORY = "oktykrk/dwell"
_LATEST_MANIFEST_URL = f"https://github.com/{_REPOSITORY}/releases/latest/download/SHA256SUMS"
_RELEASE_URL_PATTERN = re.compile(
    rf"^https://github\.com/{re.escape(_REPOSITORY)}/releases/download/"
    r"v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)/SHA256SUMS$"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class VerifiedInstaller:
    version: str
    contents: bytes


@dataclass(frozen=True)
class UpdateReport:
    previous_version: str
    installed_version: str
    restarted: bool


CommandRunner = Callable[[Sequence[str], Mapping[str, str] | None], int]
InstallerFetcher = Callable[[], VerifiedInstaller]
StatusReader = Callable[[DwellConfig], dict[str, object]]
Stopper = Callable[[DwellConfig], bool]


def _download(client: httpx.Client, url: str) -> httpx.Response:
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise DwellError("update_failed", f"Could not download {url}: {exc}") from exc
    return response


def _release_version(response: httpx.Response) -> str:
    urls = [str(item.url) for item in response.history]
    urls.append(str(response.url))
    for url in urls:
        match = _RELEASE_URL_PATTERN.fullmatch(url)
        if match is not None:
            return match.group("version")
    raise DwellError(
        "update_failed",
        "The latest release redirect did not identify a semantic Dwell version.",
    )


def _manifest_checksum(manifest: bytes, filename: str) -> str:
    matches: list[str] = []
    try:
        lines = manifest.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise DwellError("update_failed", "The release checksum manifest is not UTF-8.") from exc
    for line in lines:
        fields = line.split()
        if len(fields) == 2 and fields[1].lstrip("*") == filename:
            matches.append(fields[0])
    if len(matches) != 1 or _SHA256_PATTERN.fullmatch(matches[0]) is None:
        raise DwellError(
            "update_failed",
            f"The release checksum manifest does not contain one valid {filename} entry.",
        )
    return matches[0]


def fetch_verified_installer() -> VerifiedInstaller:
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "dwell-updater",
    }
    with httpx.Client(follow_redirects=True, timeout=30, headers=headers) as client:
        manifest_response = _download(client, _LATEST_MANIFEST_URL)
        version = _release_version(manifest_response)
        installer_url = f"https://github.com/{_REPOSITORY}/releases/download/v{version}/install.sh"
        installer = _download(client, installer_url).content

    expected_checksum = _manifest_checksum(manifest_response.content, "install.sh")
    actual_checksum = hashlib.sha256(installer).hexdigest()
    if actual_checksum != expected_checksum:
        raise DwellError("update_failed", "The downloaded installer failed SHA-256 verification.")
    return VerifiedInstaller(version=version, contents=installer)


def _run_command(command: Sequence[str], environment: Mapping[str, str] | None) -> int:
    return subprocess.run(command, env=environment, check=False).returncode  # noqa: S603


class UpdateManager:
    def __init__(
        self,
        config: DwellConfig,
        *,
        current_version: str,
        installer_fetcher: InstallerFetcher = fetch_verified_installer,
        runner: CommandRunner = _run_command,
        status_reader: StatusReader = server_status,
        stopper: Stopper = stop_server,
        launcher_resolver: Callable[[], str | None] = lambda: shutil.which("dwell"),
        reporter: Callable[[str], None] = print,
    ) -> None:
        self.config = config
        self.current_version = current_version
        self.installer_fetcher = installer_fetcher
        self.runner = runner
        self.status_reader = status_reader
        self.stopper = stopper
        self.launcher_resolver = launcher_resolver
        self.reporter = reporter

    def _run(self, command: Sequence[str], *, environment: Mapping[str, str] | None = None) -> None:
        if self.runner(command, environment) != 0:
            raise DwellError(
                "update_failed",
                f"Update command failed: {' '.join(command)}",
            )

    def run(self) -> UpdateReport:
        launcher = self.launcher_resolver()
        if launcher is None or not Path(launcher).is_absolute():
            raise DwellError(
                "update_failed",
                "Could not resolve the installed dwell launcher from PATH.",
            )

        self.reporter("Checking the latest release")
        installer = self.installer_fetcher()
        was_running = bool(self.status_reader(self.config).get("running"))
        if was_running:
            self.reporter("Stopping Dwell")
            self.stopper(self.config)

        restarted = False
        try:
            self.reporter(f"Installing Dwell {installer.version}")
            with tempfile.TemporaryDirectory(prefix="dwell-update-") as temporary_directory:
                installer_path = Path(temporary_directory) / "install.sh"
                installer_path.write_bytes(installer.contents)
                environment = dict(os.environ)
                environment["DWELL_VERSION"] = installer.version
                self._run(["/bin/sh", str(installer_path)], environment=environment)

            updated_launcher = self.launcher_resolver()
            if updated_launcher is None or not Path(updated_launcher).is_absolute():
                raise DwellError(
                    "update_failed",
                    "The updated dwell launcher could not be resolved from PATH.",
                )

            self.reporter("Updating the managed runtime")
            self._run([updated_launcher, "setup", "--upgrade", "--force"])
            if was_running:
                self.reporter("Restarting Dwell")
                self._run([updated_launcher, "start"])
                restarted = True
            self.reporter("Running diagnostics")
            self._run([updated_launcher, "doctor"])
        except Exception:
            if was_running and not restarted:
                recovery_launcher = self.launcher_resolver() or launcher
                try:
                    self._run([recovery_launcher, "start"])
                except Exception:
                    self.reporter("! Dwell could not be restarted after the failed update")
            raise

        return UpdateReport(
            previous_version=self.current_version,
            installed_version=installer.version,
            restarted=restarted,
        )
