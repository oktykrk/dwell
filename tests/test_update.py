from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import httpx
import pytest

from dwell.config import DwellConfig
from dwell.errors import DwellError
from dwell.update import UpdateManager, VerifiedInstaller, fetch_verified_installer


def test_fetch_verified_installer_pins_redirected_version_and_checksum(monkeypatch) -> None:
    installer = b"#!/bin/sh\necho installed\n"
    checksum = hashlib.sha256(installer).hexdigest()
    manifest = f"{checksum}  install.sh\n".encode()
    latest_url = "https://github.com/oktykrk/dwell/releases/latest/download/SHA256SUMS"
    versioned_manifest_url = "https://github.com/oktykrk/dwell/releases/download/v1.2.3/SHA256SUMS"
    installer_url = "https://github.com/oktykrk/dwell/releases/download/v1.2.3/install.sh"

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def get(self, url: str) -> httpx.Response:
            if url == latest_url:
                history = [
                    httpx.Response(
                        302,
                        request=httpx.Request("GET", latest_url),
                    ),
                    httpx.Response(
                        302,
                        request=httpx.Request("GET", versioned_manifest_url),
                    ),
                ]
                return httpx.Response(
                    200,
                    content=manifest,
                    history=history,
                    request=httpx.Request("GET", "https://release-assets.example/manifest"),
                )
            assert url == installer_url
            return httpx.Response(
                200,
                content=installer,
                request=httpx.Request("GET", installer_url),
            )

    monkeypatch.setattr("dwell.update.httpx.Client", FakeClient)

    payload = fetch_verified_installer()

    assert payload == VerifiedInstaller(version="1.2.3", contents=installer)


def test_fetch_verified_installer_rejects_checksum_mismatch(monkeypatch) -> None:
    manifest_url = "https://github.com/oktykrk/dwell/releases/download/v1.2.3/SHA256SUMS"

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def get(self, url: str) -> httpx.Response:
            if url.endswith("SHA256SUMS"):
                return httpx.Response(
                    200,
                    content=("0" * 64 + "  install.sh\n").encode(),
                    request=httpx.Request("GET", manifest_url),
                )
            return httpx.Response(
                200,
                content=b"installer",
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr("dwell.update.httpx.Client", FakeClient)

    with pytest.raises(DwellError, match="SHA-256 verification"):
        fetch_verified_installer()


def test_update_stops_installs_upgrades_restarts_and_diagnoses(tmp_path) -> None:
    commands: list[tuple[list[str], Mapping[str, str] | None]] = []
    stopped: list[DwellConfig] = []
    messages: list[str] = []
    launcher = "/usr/local/bin/dwell"

    def runner(command: Sequence[str], environment: Mapping[str, str] | None) -> int:
        values = list(command)
        commands.append((values, environment))
        if values[0] == "/bin/sh":
            assert open(values[1], "rb").read() == b"installer"  # noqa: PTH123, SIM115
            assert environment is not None
            assert environment["DWELL_VERSION"] == "0.2.1"
        return 0

    config = DwellConfig(home=tmp_path / "dwell")
    report = UpdateManager(
        config,
        current_version="0.2.0",
        installer_fetcher=lambda: VerifiedInstaller("0.2.1", b"installer"),
        runner=runner,
        status_reader=lambda _config: {"running": True},
        stopper=lambda value: stopped.append(value) is None,
        launcher_resolver=lambda: launcher,
        reporter=messages.append,
    ).run()

    assert stopped == [config]
    assert [command for command, _environment in commands] == [
        ["/bin/sh", commands[0][0][1]],
        [launcher, "setup", "--upgrade", "--force"],
        [launcher, "start"],
        [launcher, "doctor"],
    ]
    assert report.previous_version == "0.2.0"
    assert report.installed_version == "0.2.1"
    assert report.restarted is True
    assert messages == [
        "Checking the latest release",
        "Stopping Dwell",
        "Installing Dwell 0.2.1",
        "Updating the managed runtime",
        "Restarting Dwell",
        "Running diagnostics",
    ]


def test_update_restores_running_service_after_setup_failure(tmp_path) -> None:
    commands: list[list[str]] = []
    launcher = "/usr/local/bin/dwell"

    def runner(command: Sequence[str], _environment: Mapping[str, str] | None) -> int:
        values = list(command)
        commands.append(values)
        if values[1:] == ["setup", "--upgrade", "--force"]:
            return 1
        return 0

    manager = UpdateManager(
        DwellConfig(home=tmp_path / "dwell"),
        current_version="0.2.0",
        installer_fetcher=lambda: VerifiedInstaller("0.2.1", b"installer"),
        runner=runner,
        status_reader=lambda _config: {"running": True},
        stopper=lambda _config: True,
        launcher_resolver=lambda: launcher,
        reporter=lambda _message: None,
    )

    with pytest.raises(DwellError, match="setup --upgrade --force"):
        manager.run()

    assert commands[-1] == [launcher, "start"]
