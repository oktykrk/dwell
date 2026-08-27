from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import tarfile
import textwrap
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "install.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _executable(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _write_zip(path: Path, member: str, source: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        info = zipfile.ZipInfo(member)
        info.external_attr = 0o100755 << 16
        archive.writestr(info, source.read_bytes())


def _fake_uv(path: Path) -> Path:
    return _executable(
        path,
        r"""
        #!/bin/sh
        set -eu
        if [ "${1:-}" = "--version" ]; then
            echo "uv 0.11.32 (3010295ae 2026-07-23 aarch64-apple-darwin)"
            exit 0
        fi
        test -z "${UV_PYTHON_DOWNLOADS_JSON_URL:-}"
        test -z "${UV_PYTHON_INSTALL_MIRROR:-}"
        test -z "${UV_NO_VERIFY_HASHES:-}"
        test -z "${PIP_CONFIG_FILE:-}"
        quiet=""
        if [ "${1:-}" = "--quiet" ]; then
            quiet="--quiet"
            shift
        fi
        if [ "${1:-}" = "--no-config" ]; then
            shift
        fi
        printf '%s %s\n' "$quiet" "$*" >>"$DWELL_TEST_UV_LOG"
        command=$1
        shift
        case "$command" in
            python)
                test "$1" = install
                mkdir -p "$UV_PYTHON_INSTALL_DIR"
                ;;
            venv)
                environment=$1
                mkdir -p "$environment/bin"
                printf '#!/bin/sh\nexit 0\n' >"$environment/bin/python"
                chmod 0755 "$environment/bin/python"
                ;;
            pip)
                operation=$1
                shift
                python=""
                wheel=""
                while [ "$#" -gt 0 ]; do
                    case "$1" in
                        --python)
                            python=$2
                            shift 2
                            ;;
                        *.whl)
                            wheel=$1
                            shift
                            ;;
                        *) shift ;;
                    esac
                done
                if [ "$operation" = install ]; then
                    environment=${python%/bin/python}
                    filename=${wheel##*/}
                    version=${filename#dwell_ai-}
                    version=${version%-py3-none-any.whl}
                    cat >"$environment/bin/dwell" <<EOF
        #!/bin/sh
        if [ "\${1:-}" = "--tools" ]; then
            command -v uv
            command -v ffmpeg
            command -v ffprobe
            exit 0
        fi
        if [ "\${DWELL_TEST_FAIL_MANAGED_LAUNCHER:-0}" = 1 ] \
            && command -v uv | grep -Fq '/tools/uv-'; then
            exit 75
        fi
        echo "Dwell $version"
        EOF
                    chmod 0755 "$environment/bin/dwell"
                fi
                ;;
            *) exit 64 ;;
        esac
        """,
    )


def _fake_curl(path: Path) -> Path:
    return _executable(
        path,
        r"""
        #!/bin/sh
        set -eu
        destination=""
        url=""
        while [ "$#" -gt 0 ]; do
            case "$1" in
                --output)
                    destination=$2
                    shift 2
                    ;;
                --proto|--retry|--retry-delay)
                    shift 2
                    ;;
                --tlsv1.2|--fail|--location|--silent|--show-error)
                    shift
                    ;;
                *)
                    url=$1
                    shift
                    ;;
            esac
        done
        test -n "$destination"
        filename=${url##*/}
        cp "$DWELL_TEST_ASSET_DIR/$filename" "$destination"
        """,
    )


def _fake_codesign(path: Path) -> Path:
    return _executable(
        path,
        r"""
        #!/bin/sh
        set -eu
        case " $* " in
            *" -dv "*)
                echo "Authority=Developer ID Application: Martin Riedl (KU3N25YGLU)" >&2
                echo "TeamIdentifier=KU3N25YGLU" >&2
                ;;
        esac
        """,
    )


def _fake_file(path: Path) -> Path:
    return _executable(
        path,
        """
        #!/bin/sh
        echo "$1: Mach-O 64-bit executable arm64"
        """,
    )


class InstallFixture:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.assets = tmp_path / "assets"
        self.fake_bin = tmp_path / "fake-bin"
        self.home = tmp_path / "home"
        self.install_root = self.home / ".local" / "share" / "dwell"
        self.bin_dir = tmp_path / "command-bin"
        self.uv_log = tmp_path / "uv.log"
        for directory in (self.assets, self.fake_bin, self.home, self.bin_dir):
            directory.mkdir(parents=True)

        self.curl = _fake_curl(self.fake_bin / "curl")
        self.codesign = _fake_codesign(self.fake_bin / "codesign")
        self.file = _fake_file(self.fake_bin / "file")
        self.uv_source = _fake_uv(tmp_path / "uv")
        self.ffmpeg_source = _executable(
            tmp_path / "ffmpeg", '#!/bin/sh\necho "ffmpeg version 9.0.1-test"\n'
        )
        self.ffprobe_source = _executable(
            tmp_path / "ffprobe", '#!/bin/sh\necho "ffprobe version 9.0.1-test"\n'
        )
        self.uv_archive = self.assets / "uv-aarch64-apple-darwin.tar.gz"
        with tarfile.open(self.uv_archive, "w:gz") as archive:
            directory = tarfile.TarInfo("uv-aarch64-apple-darwin/")
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o755
            archive.addfile(directory)
            archive.add(self.uv_source, "uv-aarch64-apple-darwin/uvx")
            archive.add(self.uv_source, "uv-aarch64-apple-darwin/uv")

        self.ffmpeg_archive = self.assets / "ffmpeg.zip"
        self.ffprobe_archive = self.assets / "ffprobe.zip"
        _write_zip(self.ffmpeg_archive, "ffmpeg", self.ffmpeg_source)
        _write_zip(self.ffprobe_archive, "ffprobe", self.ffprobe_source)
        self.write_release("0.1.0")

    def write_release(self, version: str, *, corrupt_wheel: bool = False) -> None:
        for old_wheel in self.assets.glob("dwell_ai-*-py3-none-any.whl"):
            old_wheel.unlink()
        self.wheel = self.assets / f"dwell_ai-{version}-py3-none-any.whl"
        self.wheel.write_bytes(f"fake wheel {version}\n".encode())
        self.requirements = self.assets / "requirements-macos-arm64-py311.txt"
        self.requirements.write_text("demo==1.0 --hash=sha256:" + "0" * 64 + "\n", encoding="utf-8")
        wheel_sha = "f" * 64 if corrupt_wheel else _sha256(self.wheel)
        (self.assets / "SHA256SUMS").write_text(
            f"{wheel_sha}  {self.wheel.name}\n"
            f"{_sha256(self.requirements)}  {self.requirements.name}\n",
            encoding="utf-8",
        )

    def environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "HOME": str(self.home),
            "PATH": f"{self.bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin",
            "DWELL_INSTALLER_TEST_MODE": "1",
            "DWELL_INSTALL_ROOT": str(self.install_root),
            "DWELL_BIN_DIR": str(self.bin_dir),
            "DWELL_RELEASE_BASE_URL": "https://example.invalid/release",
            "DWELL_TEST_SYSTEM": "Darwin",
            "DWELL_TEST_MACHINE": "arm64",
            "DWELL_TEST_MACOS_VERSION": "15.0",
            "DWELL_TEST_CURL": str(self.curl),
            "DWELL_TEST_CODESIGN": str(self.codesign),
            "DWELL_TEST_FILE": str(self.file),
            "DWELL_TEST_ASSET_DIR": str(self.assets),
            "DWELL_TEST_UV_LOG": str(self.uv_log),
            "DWELL_TEST_UV_ARCHIVE_URL": "https://example.invalid/uv-aarch64-apple-darwin.tar.gz",
            "DWELL_TEST_UV_ARCHIVE_SHA256": _sha256(self.uv_archive),
            "DWELL_TEST_UV_BINARY_SHA256": _sha256(self.uv_source),
            "DWELL_TEST_FFMPEG_URL": "https://example.invalid/ffmpeg.zip",
            "DWELL_TEST_FFMPEG_ARCHIVE_SHA256": _sha256(self.ffmpeg_archive),
            "DWELL_TEST_FFMPEG_BINARY_SHA256": _sha256(self.ffmpeg_source),
            "DWELL_TEST_FFPROBE_URL": "https://example.invalid/ffprobe.zip",
            "DWELL_TEST_FFPROBE_ARCHIVE_SHA256": _sha256(self.ffprobe_archive),
            "DWELL_TEST_FFPROBE_BINARY_SHA256": _sha256(self.ffprobe_source),
        }

    def launcher_environment(self) -> dict[str, str]:
        environment = self.environment()
        environment.pop("DWELL_INSTALL_ROOT")
        return environment

    def install(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        environment = self.environment()
        environment.update(overrides)
        return subprocess.run(
            ["/bin/sh", str(INSTALLER)],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )


@pytest.fixture
def install_fixture(tmp_path: Path) -> InstallFixture:
    return InstallFixture(tmp_path)


def test_installs_self_contained_cli_without_source_build(
    install_fixture: InstallFixture,
) -> None:
    result = install_fixture.install()

    assert result.returncode == 0, result.stderr
    assert "Dwell 0.1.0 installed successfully." in result.stdout
    launcher = install_fixture.bin_dir / "dwell"
    assert launcher.is_file()
    assert os.access(launcher, os.X_OK)
    current = install_fixture.install_root / "current"
    assert current.is_symlink()

    command = subprocess.run(
        [str(launcher), "--tools"],
        env=install_fixture.launcher_environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    tool_paths = command.stdout.splitlines()
    assert tool_paths == [
        str(install_fixture.install_root / "tools" / "uv-0.11.32" / "bin" / "uv"),
        str(install_fixture.install_root / "tools" / "ffmpeg-9.0.1" / "bin" / "ffmpeg"),
        str(install_fixture.install_root / "tools" / "ffmpeg-9.0.1" / "bin" / "ffprobe"),
    ]

    uv_log = install_fixture.uv_log.read_text(encoding="utf-8")
    assert uv_log.count("--quiet ") == 4
    assert "python install 3.11.15 --no-bin" in uv_log
    assert "--require-hashes --no-build --strict" in uv_log
    assert "--managed-python --no-deps --no-build" in uv_log
    assert "--no-build" in uv_log


def test_failed_upgrade_does_not_change_current_release(
    install_fixture: InstallFixture,
) -> None:
    first = install_fixture.install()
    assert first.returncode == 0, first.stderr
    current = install_fixture.install_root / "current"
    original_target = current.readlink()

    install_fixture.write_release("0.2.0", corrupt_wheel=True)
    failed = install_fixture.install()

    assert failed.returncode != 0
    assert "SHA-256 mismatch" in failed.stderr
    assert current.readlink() == original_target
    launcher = install_fixture.bin_dir / "dwell"
    command = subprocess.run(
        [str(launcher), "--version"],
        env=install_fixture.launcher_environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert command.stdout.strip() == "Dwell 0.1.0"


def test_upgrade_switches_atomically_and_preserves_previous_release(
    install_fixture: InstallFixture,
) -> None:
    first = install_fixture.install()
    assert first.returncode == 0, first.stderr
    current = install_fixture.install_root / "current"
    original_target = current.readlink()

    install_fixture.write_release("0.2.0")
    second = install_fixture.install()

    assert second.returncode == 0, second.stderr
    assert current.readlink() != original_target
    assert original_target.is_dir()
    launcher = install_fixture.bin_dir / "dwell"
    command = subprocess.run(
        [str(launcher), "--version"],
        env=install_fixture.launcher_environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert command.stdout.strip() == "Dwell 0.2.0"


def test_reinstalling_same_release_is_idempotent(install_fixture: InstallFixture) -> None:
    first = install_fixture.install()
    assert first.returncode == 0, first.stderr
    current = install_fixture.install_root / "current"
    original_target = current.readlink()
    release_count = len(list((install_fixture.install_root / "releases").iterdir()))
    original_uv_log = install_fixture.uv_log.read_text(encoding="utf-8")

    second = install_fixture.install()

    assert second.returncode == 0, second.stderr
    assert current.readlink() == original_target
    assert len(list((install_fixture.install_root / "releases").iterdir())) == release_count
    assert install_fixture.uv_log.read_text(encoding="utf-8") == original_uv_log


def test_refuses_to_overwrite_foreign_launcher(install_fixture: InstallFixture) -> None:
    launcher = install_fixture.bin_dir / "dwell"
    launcher.write_text("foreign command\n", encoding="utf-8")

    result = install_fixture.install()

    assert result.returncode != 0
    assert "refusing to overwrite existing command" in result.stderr
    assert launcher.read_text(encoding="utf-8") == "foreign command\n"
    assert not install_fixture.install_root.exists()


def test_rejects_unsupported_platform_before_downloading(
    install_fixture: InstallFixture,
) -> None:
    result = install_fixture.install(DWELL_TEST_MACHINE="x86_64")

    assert result.returncode != 0
    assert "requires Apple Silicon" in result.stderr
    assert not install_fixture.install_root.exists()


def test_scrubs_all_uv_and_pip_overrides(install_fixture: InstallFixture) -> None:
    result = install_fixture.install(
        UV_PYTHON_DOWNLOADS_JSON_URL="https://example.invalid/python.json",
        UV_PYTHON_INSTALL_MIRROR="https://example.invalid/python",
        UV_NO_VERIFY_HASHES="1",
        PIP_CONFIG_FILE="/tmp/hostile-pip.conf",
    )

    assert result.returncode == 0, result.stderr


def test_recovers_a_stale_install_lock(install_fixture: InstallFixture) -> None:
    lock = install_fixture.install_root / ".install-lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text("2147483647\n", encoding="utf-8")

    result = install_fixture.install()

    assert result.returncode == 0, result.stderr
    assert not lock.exists()


def test_refuses_a_shadowing_dwell_command(install_fixture: InstallFixture) -> None:
    shadow_bin = install_fixture.root / "shadow-bin"
    shadow_bin.mkdir()
    shadow = _executable(shadow_bin / "dwell", "#!/bin/sh\nexit 0\n")

    result = install_fixture.install(
        PATH=(f"{shadow_bin}:{install_fixture.bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin")
    )

    assert result.returncode != 0
    assert str(shadow) in result.stderr
    assert "would shadow this install" in result.stderr
    assert not install_fixture.install_root.exists()


def test_test_hooks_are_rejected_outside_explicit_test_mode(
    install_fixture: InstallFixture,
) -> None:
    result = install_fixture.install(DWELL_INSTALLER_TEST_MODE="0")

    assert result.returncode != 0
    assert "DWELL_TEST_* overrides require DWELL_INSTALLER_TEST_MODE=1" in result.stderr
    assert not install_fixture.install_root.exists()


def test_failed_post_switch_smoke_restores_previous_release(
    install_fixture: InstallFixture,
) -> None:
    first = install_fixture.install()
    assert first.returncode == 0, first.stderr
    current = install_fixture.install_root / "current"
    original_target = current.readlink()
    original_release_count = len(list((install_fixture.install_root / "releases").iterdir()))

    install_fixture.write_release("0.2.0")
    failed = install_fixture.install(DWELL_TEST_FAIL_MANAGED_LAUNCHER="1")

    assert failed.returncode != 0
    assert "installed launcher failed its version smoke test" in failed.stderr
    assert current.readlink() == original_target
    assert len(list((install_fixture.install_root / "releases").iterdir())) == (
        original_release_count
    )
    launcher = install_fixture.bin_dir / "dwell"
    command = subprocess.run(
        [str(launcher), "--version"],
        env=install_fixture.launcher_environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert command.stdout.strip() == "Dwell 0.1.0"


def test_launcher_bakes_custom_install_root(install_fixture: InstallFixture) -> None:
    custom_root = install_fixture.home / "custom'root"

    result = install_fixture.install(DWELL_INSTALL_ROOT=str(custom_root))

    assert result.returncode == 0, result.stderr
    launcher = install_fixture.bin_dir / "dwell"
    environment = install_fixture.launcher_environment()
    environment.pop("DWELL_INSTALL_ROOT", None)
    command = subprocess.run(
        [str(launcher), "--version"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert command.stdout.strip() == "Dwell 0.1.0"
    assert custom_root.is_dir()
