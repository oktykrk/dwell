#!/bin/sh

set -eu

# Dwell's installer is intentionally self-contained. uv is an internal install
# engine; neither uv nor its managed Python is added to the user's PATH.
UV_VERSION="0.11.32"
UV_PYTHON_VERSION="3.11.15"
UV_ARCHIVE_URL_DEFAULT="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-aarch64-apple-darwin.tar.gz"
UV_ARCHIVE_SHA256_DEFAULT="ed336d0ba49db8ef89b2b41fffa372ce63bd032f22a56f001c265891aec32829"
UV_BINARY_SHA256_DEFAULT="3736babdf838efb1c04ca690dd6ff3458a23cdf98e0e08b1f721eac4779e272d"

FFMPEG_VERSION="9.0.1"
FFMPEG_URL_DEFAULT="https://ffmpeg.martin-riedl.de/download/macos/arm64/1787073674_9.0.1/ffmpeg.zip"
FFMPEG_ARCHIVE_SHA256_DEFAULT="8287a1b2229e05eb41859f073e18e6c52c60a778f2f5e6881070fe51b79407fe"
FFMPEG_BINARY_SHA256_DEFAULT="393e4c395020a1cb7cbd77fbe00599ce69d1c6466fee0dbd59d13f86a81a1611"
FFPROBE_URL_DEFAULT="https://ffmpeg.martin-riedl.de/download/macos/arm64/1787073674_9.0.1/ffprobe.zip"
FFPROBE_ARCHIVE_SHA256_DEFAULT="102a26b8940a053298d9929bfaae71e4b6ef65ba5f19a99a88c433108560741a"
FFPROBE_BINARY_SHA256_DEFAULT="7abc49fb2bdf2204f018e76dc6e0a8ae7643313bae09a9fa43e7eb12442271bc"
FFMPEG_TEAM_ID="KU3N25YGLU"

REPOSITORY="oktykrk/dwell"
REQUIREMENTS_NAME="requirements-macos-arm64-py311.txt"

say() {
    printf '%s\n' "$*"
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

is_sha256() {
    printf '%s\n' "$1" | LC_ALL=C grep -Eq '^[0-9a-f]{64}$'
}

is_release_version() {
    printf '%s\n' "$1" | LC_ALL=C grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'
}

sha256_file() {
    "$SHASUM" -a 256 "$1" | awk '{ print $1 }'
}

verify_sha256() {
    file=$1
    expected=$2
    label=$3
    is_sha256 "$expected" || die "invalid pinned SHA-256 for $label"
    actual=$(sha256_file "$file")
    [ "$actual" = "$expected" ] || die "SHA-256 mismatch for $label"
}

download() {
    url=$1
    destination=$2
    partial="${destination}.part"
    "$CURL" \
        --proto '=https' \
        --tlsv1.2 \
        --fail \
        --location \
        --silent \
        --show-error \
        --retry 3 \
        --retry-delay 1 \
        --output "$partial" \
        "$url"
    mv -f "$partial" "$destination"
}

checksum_from_manifest() {
    filename=$1
    manifest=$2
    count=$(awk -v name="$filename" '$2 == name { count += 1 } END { print count + 0 }' "$manifest")
    [ "$count" -eq 1 ] || die "SHA256SUMS must contain exactly one entry for $filename"
    checksum=$(awk -v name="$filename" '$2 == name { print $1 }' "$manifest")
    is_sha256 "$checksum" || die "SHA256SUMS contains an invalid digest for $filename"
    printf '%s\n' "$checksum"
}

validate_path() {
    label=$1
    path=$2
    case "$path" in
        /*) ;;
        *) die "$label must be an absolute path: $path" ;;
    esac
    [ "$path" != "/" ] || die "$label may not be the filesystem root"
    [ "$(printf '%s' "$path" | wc -l | tr -d ' ')" -eq 0 ] \
        || die "$label may not contain a newline"
    case "$path" in
        *:*) die "$label may not contain a colon: $path" ;;
    esac
}

path_contains_directory() {
    directory=$1
    case ":${PATH:-}:" in
        *":$directory:"*) return 0 ;;
        *) return 1 ;;
    esac
}

sanitize_package_environment() {
    # uv has many environment-variable controls, including mirrors and index
    # overrides. Remove the whole UV_/PIP_ namespaces instead of maintaining a
    # deny-list that will become incomplete as uv and pip add new settings.
    package_environment_names=$(
        env | sed -n 's/^\(UV_[A-Za-z0-9_]*\)=.*/\1/p; s/^\(PIP_[A-Za-z0-9_]*\)=.*/\1/p'
    )
    for package_environment_name in $package_environment_names; do
        unset "$package_environment_name"
    done
    unset VIRTUAL_ENV PYTHONHOME PYTHONPATH
}

validate_platform() {
    system=${DWELL_TEST_SYSTEM:-$($UNAME_CMD -s)}
    machine=${DWELL_TEST_MACHINE:-$($UNAME_CMD -m)}
    [ "$system" = "Darwin" ] || die "Dwell requires macOS; detected $system"
    [ "$machine" = "arm64" ] || die "Dwell requires Apple Silicon; detected $machine"

    if [ -n "${DWELL_TEST_MACOS_VERSION:-}" ]; then
        macos_version=$DWELL_TEST_MACOS_VERSION
    else
        require_command "$SW_VERS"
        macos_version=$($SW_VERS -productVersion)
    fi
    macos_major=${macos_version%%.*}
    case "$macos_major" in
        ''|*[!0-9]*) die "unable to determine the macOS version: $macos_version" ;;
    esac
    [ "$macos_major" -ge 14 ] || die "Dwell requires macOS 14 or newer; detected $macos_version"
}

validate_macho_arm64() {
    binary=$1
    label=$2
    description=$($FILE_CMD "$binary")
    case "$description" in
        *"Mach-O 64-bit executable arm64"*) ;;
        *) die "$label is not a native arm64 Mach-O executable" ;;
    esac
}

validate_uv_binary() {
    binary=$1
    [ -f "$binary" ] && [ -x "$binary" ] || return 1
    [ "$(sha256_file "$binary")" = "$UV_BINARY_SHA256" ] || return 1
    validate_macho_arm64 "$binary" "uv"
    [ "$("$binary" --version)" = "uv $UV_VERSION (3010295ae 2026-07-23 aarch64-apple-darwin)" ] || return 1
}

validate_ffmpeg_binary() {
    binary=$1
    name=$2
    expected_sha=$3
    [ -f "$binary" ] && [ -x "$binary" ] || return 1
    [ "$(sha256_file "$binary")" = "$expected_sha" ] || return 1
    validate_macho_arm64 "$binary" "$name"
    "$CODESIGN" --verify --strict "$binary" >/dev/null 2>&1 || return 1
    signature=$($CODESIGN -dv --verbose=4 "$binary" 2>&1) || return 1
    printf '%s\n' "$signature" | grep -Fqx "TeamIdentifier=$FFMPEG_TEAM_ID" || return 1
    printf '%s\n' "$signature" \
        | grep -Fqx "Authority=Developer ID Application: Martin Riedl ($FFMPEG_TEAM_ID)" \
        || return 1
    first_line=$("$binary" -version 2>/dev/null | sed -n '1p') || return 1
    case "$first_line" in
        "$name version $FFMPEG_VERSION-"*) ;;
        *) return 1 ;;
    esac
}

prepare_uv() {
    uv_dir="$tools_root/uv-$UV_VERSION"
    uv_binary="$uv_dir/bin/uv"
    if [ -e "$uv_dir" ] || [ -L "$uv_dir" ]; then
        validate_uv_binary "$uv_binary" \
            || die "managed uv installation is invalid: $uv_dir"
        UV_BIN=$uv_binary
        return
    fi

    archive="$temporary_root/uv-aarch64-apple-darwin.tar.gz"
    extracted="$temporary_root/uv-extracted"
    listing="$temporary_root/uv-archive.list"
    download "$UV_ARCHIVE_URL" "$archive"
    verify_sha256 "$archive" "$UV_ARCHIVE_SHA256" "uv archive"
    "$TAR" -tzf "$archive" >"$listing"
    [ "$(wc -l <"$listing" | tr -d ' ')" -eq 3 ] \
        || die "uv archive has an unexpected number of entries"
    grep -Fqx 'uv-aarch64-apple-darwin/' "$listing" \
        || die "uv archive is missing its root directory"
    grep -Fqx 'uv-aarch64-apple-darwin/uv' "$listing" \
        || die "uv archive is missing uv"
    grep -Fqx 'uv-aarch64-apple-darwin/uvx' "$listing" \
        || die "uv archive is missing uvx"
    mkdir -p "$extracted"
    "$TAR" -xzf "$archive" -C "$extracted"
    extracted_uv="$extracted/uv-aarch64-apple-darwin/uv"
    chmod 0755 "$extracted_uv"
    validate_uv_binary "$extracted_uv" || die "downloaded uv binary failed validation"

    uv_staging="$tools_root/.uv-$UV_VERSION.installing-$$"
    [ ! -e "$uv_staging" ] && [ ! -L "$uv_staging" ] \
        || die "temporary uv install path already exists: $uv_staging"
    mkdir -p "$uv_staging/bin"
    "$INSTALL_CMD" -m 0755 "$extracted_uv" "$uv_staging/bin/uv"
    mv "$uv_staging" "$uv_dir"
    uv_staging=""
    UV_BIN=$uv_binary
}

zip_has_only() {
    archive=$1
    expected=$2
    listing=$3
    "$UNZIP" -Z1 "$archive" >"$listing"
    [ "$(wc -l <"$listing" | tr -d ' ')" -eq 1 ] || return 1
    grep -Fqx "$expected" "$listing"
}

prepare_ffmpeg() {
    ffmpeg_dir="$tools_root/ffmpeg-$FFMPEG_VERSION"
    ffmpeg_binary="$ffmpeg_dir/bin/ffmpeg"
    ffprobe_binary="$ffmpeg_dir/bin/ffprobe"
    if [ -e "$ffmpeg_dir" ] || [ -L "$ffmpeg_dir" ]; then
        validate_ffmpeg_binary "$ffmpeg_binary" ffmpeg "$FFMPEG_BINARY_SHA256" \
            || die "managed ffmpeg installation is invalid: $ffmpeg_dir"
        validate_ffmpeg_binary "$ffprobe_binary" ffprobe "$FFPROBE_BINARY_SHA256" \
            || die "managed ffprobe installation is invalid: $ffmpeg_dir"
        FFMPEG_BIN_DIR="$ffmpeg_dir/bin"
        return
    fi

    ffmpeg_archive="$temporary_root/ffmpeg.zip"
    ffprobe_archive="$temporary_root/ffprobe.zip"
    ffmpeg_extract="$temporary_root/ffmpeg-extracted"
    ffprobe_extract="$temporary_root/ffprobe-extracted"
    download "$FFMPEG_URL" "$ffmpeg_archive"
    download "$FFPROBE_URL" "$ffprobe_archive"
    verify_sha256 "$ffmpeg_archive" "$FFMPEG_ARCHIVE_SHA256" "ffmpeg archive"
    verify_sha256 "$ffprobe_archive" "$FFPROBE_ARCHIVE_SHA256" "ffprobe archive"
    zip_has_only "$ffmpeg_archive" ffmpeg "$temporary_root/ffmpeg-archive.list" \
        || die "ffmpeg archive must contain exactly one member named ffmpeg"
    zip_has_only "$ffprobe_archive" ffprobe "$temporary_root/ffprobe-archive.list" \
        || die "ffprobe archive must contain exactly one member named ffprobe"
    mkdir -p "$ffmpeg_extract" "$ffprobe_extract"
    "$UNZIP" -qq "$ffmpeg_archive" -d "$ffmpeg_extract"
    "$UNZIP" -qq "$ffprobe_archive" -d "$ffprobe_extract"
    chmod 0755 "$ffmpeg_extract/ffmpeg" "$ffprobe_extract/ffprobe"
    verify_sha256 "$ffmpeg_extract/ffmpeg" "$FFMPEG_BINARY_SHA256" "ffmpeg binary"
    verify_sha256 "$ffprobe_extract/ffprobe" "$FFPROBE_BINARY_SHA256" "ffprobe binary"
    validate_ffmpeg_binary "$ffmpeg_extract/ffmpeg" ffmpeg "$FFMPEG_BINARY_SHA256" \
        || die "downloaded ffmpeg binary failed signature, architecture, or version validation"
    validate_ffmpeg_binary "$ffprobe_extract/ffprobe" ffprobe "$FFPROBE_BINARY_SHA256" \
        || die "downloaded ffprobe binary failed signature, architecture, or version validation"

    ffmpeg_staging="$tools_root/.ffmpeg-$FFMPEG_VERSION.installing-$$"
    [ ! -e "$ffmpeg_staging" ] && [ ! -L "$ffmpeg_staging" ] \
        || die "temporary ffmpeg install path already exists: $ffmpeg_staging"
    mkdir -p "$ffmpeg_staging/bin"
    "$INSTALL_CMD" -m 0755 "$ffmpeg_extract/ffmpeg" "$ffmpeg_staging/bin/ffmpeg"
    "$INSTALL_CMD" -m 0755 "$ffprobe_extract/ffprobe" "$ffmpeg_staging/bin/ffprobe"
    mv "$ffmpeg_staging" "$ffmpeg_dir"
    ffmpeg_staging=""
    FFMPEG_BIN_DIR="$ffmpeg_dir/bin"
}

shell_quote() {
    printf "'"
    printf '%s' "$1" | sed "s/'/'\\\\''/g"
    printf "'"
}

write_launcher() {
    launcher=$1
    root_literal=$(shell_quote "$install_root")
    cat >"$launcher" <<EOF
#!/bin/sh
# Managed by the Dwell installer. Do not edit this file in place.
set -eu
dwell_root=$root_literal
export PATH="\$dwell_root/tools/uv-$UV_VERSION/bin:\$dwell_root/tools/ffmpeg-$FFMPEG_VERSION/bin:\$PATH"
exec "\$dwell_root/current/bin/dwell" "\$@"
EOF
    chmod 0755 "$launcher"
}

launcher_is_managed() {
    target=$1
    [ -f "$target" ] || return 1
    grep -Fq '# Managed by the Dwell installer.' "$target"
}

validate_launcher_target() {
    target=$1
    if [ -e "$target" ] || [ -L "$target" ]; then
        [ ! -L "$target" ] \
            || die "refusing to overwrite launcher symlink: $target"
        launcher_is_managed "$target" \
            || die "refusing to overwrite existing command: $target"
    fi
}

install_launcher_atomically() {
    launcher=$1
    target=$2
    launcher_stage="$bin_dir/.dwell-launcher.installing-$$"
    if [ -e "$launcher_stage" ] || [ -L "$launcher_stage" ]; then
        return 1
    fi
    if [ "$use_sudo" -eq 1 ]; then
        /usr/bin/sudo /usr/bin/install -m 0755 "$launcher" "$launcher_stage" \
            || return 1
        /usr/bin/sudo /bin/mv -f "$launcher_stage" "$target" || return 1
    else
        "$INSTALL_CMD" -m 0755 "$launcher" "$launcher_stage" || return 1
        mv -f "$launcher_stage" "$target" || return 1
    fi
    launcher_stage=""
}

remove_launcher() {
    target=$1
    if [ "$use_sudo" -eq 1 ]; then
        /usr/bin/sudo /bin/rm -f "$target"
    else
        rm -f "$target"
    fi
}

replace_symlink_atomically() {
    source=$1
    destination=$2
    # BSD mv follows a destination symlink to a directory unless -h is used.
    # GNU mv uses -T for the same no-target-directory behavior.
    if mv -f -h "$source" "$destination" 2>/dev/null; then
        return
    fi
    if mv -f -T "$source" "$destination" 2>/dev/null; then
        return
    fi
    die "unable to replace the current release symlink atomically"
}

restore_previous_current() {
    [ "${current_switched:-0}" -eq 1 ] || return 0
    if [ "${previous_current_present:-0}" -eq 1 ]; then
        rollback_link="$install_root/.current-rollback-$$"
        if [ -L "$rollback_link" ]; then
            rm -f "$rollback_link"
        elif [ -e "$rollback_link" ]; then
            return 1
        fi
        ln -s "$previous_current_target" "$rollback_link" || return 1
        if mv -f -h "$rollback_link" "$current" 2>/dev/null \
            || mv -f -T "$rollback_link" "$current" 2>/dev/null; then
            current_switched=0
            return 0
        fi
        return 1
    fi
    if [ -L "$current" ]; then
        rm -f "$current" || return 1
    elif [ -e "$current" ]; then
        return 1
    fi
    current_switched=0
}

restore_previous_launcher() {
    [ "${launcher_changed:-0}" -eq 1 ] || return 0
    if [ -n "${launcher_stage:-}" ]; then
        case "$launcher_stage" in
            "$bin_dir"/.dwell-launcher.installing-*)
                if [ "${use_sudo:-0}" -eq 1 ]; then
                    /usr/bin/sudo /bin/rm -f "$launcher_stage" || return 1
                else
                    rm -f "$launcher_stage" || return 1
                fi
                launcher_stage=""
                ;;
            *) return 1 ;;
        esac
    fi
    if [ "${previous_launcher_present:-0}" -eq 1 ]; then
        install_launcher_atomically "$previous_launcher" "$launcher_target" || return 1
    else
        remove_launcher "$launcher_target" || return 1
    fi
    launcher_changed=0
}

acquire_install_lock() {
    if mkdir "$install_lock" 2>/dev/null; then
        :
    elif [ -d "$install_lock" ] && [ ! -L "$install_lock" ]; then
        stale_pid=""
        if [ -f "$install_lock/pid" ] && [ ! -L "$install_lock/pid" ]; then
            stale_pid=$(cat "$install_lock/pid" 2>/dev/null || true)
        fi
        case "$stale_pid" in
            ''|*[!0-9]*)
                die "install lock has no valid owner PID; inspect and remove it if no install is running: $install_lock"
                ;;
            *)
                if kill -0 "$stale_pid" 2>/dev/null; then
                    lock_is_live=1
                else
                    lock_is_live=0
                fi
                ;;
        esac
        [ "$lock_is_live" -eq 0 ] \
            || die "another Dwell install is running with PID $stale_pid"
        rm -f "$install_lock/pid"
        rmdir "$install_lock" 2>/dev/null \
            || die "stale install lock is not safe to remove: $install_lock"
        mkdir "$install_lock" 2>/dev/null \
            || die "another Dwell install started concurrently"
    else
        die "install lock is not a safe directory: $install_lock"
    fi
    printf '%s\n' "$$" >"$install_lock/pid"
    lock_owned=1
}

cleanup() {
    if [ "${install_succeeded:-0}" -ne 1 ]; then
        restore_previous_current \
            || printf 'error: failed to restore the previous Dwell release\n' >&2
        restore_previous_launcher \
            || printf 'error: failed to restore the previous Dwell launcher\n' >&2
    fi
    if [ -n "${release_created:-}" ] && [ "$release_created" -eq 1 ]; then
        release_is_current=0
        if [ -n "${current:-}" ] && [ -L "$current" ] \
            && [ "$(readlink "$current" 2>/dev/null || true)" = "${release_dir:-}" ]; then
            release_is_current=1
        fi
        if [ "$release_is_current" -eq 0 ]; then
            case "${release_dir:-}" in
                "$releases_dir"/*) rm -rf "$release_dir" || true ;;
            esac
        else
            printf 'error: preserving the new release because it is still active\n' >&2
        fi
    fi
    for temporary_link in "${new_current:-}" "${rollback_link:-}"; do
        [ -n "$temporary_link" ] || continue
        case "$temporary_link" in
            "$install_root"/.current-*)
                [ ! -L "$temporary_link" ] || rm -f "$temporary_link" || true
                ;;
        esac
    done
    for tool_staging in "${uv_staging:-}" "${ffmpeg_staging:-}"; do
        [ -n "$tool_staging" ] || continue
        case "$tool_staging" in
            "$tools_root"/.uv-*.installing-*|"$tools_root"/.ffmpeg-*.installing-*)
                rm -rf "$tool_staging" || true
                ;;
        esac
    done
    if [ -n "${launcher_stage:-}" ]; then
        case "$launcher_stage" in
            "$bin_dir"/.dwell-launcher.installing-*)
                if [ "${use_sudo:-0}" -eq 1 ]; then
                    /usr/bin/sudo /bin/rm -f "$launcher_stage" 2>/dev/null || true
                else
                    rm -f "$launcher_stage" || true
                fi
                ;;
        esac
    fi
    if [ -n "${lock_owned:-}" ] && [ "$lock_owned" -eq 1 ]; then
        rm -f "$install_lock/pid" || true
        rmdir "$install_lock" 2>/dev/null || true
    fi
    if [ -n "${temporary_root:-}" ]; then
        case "$temporary_root" in
            "${TMPDIR:-/tmp}"/dwell-install.*) rm -rf "$temporary_root" || true ;;
        esac
    fi
}

trap cleanup 0
trap 'exit 1' HUP INT TERM

umask 022

# This check is deliberately outside the test-hook mechanism. No environment
# override may make a root-run installer execute test commands with privileges.
[ "$(/usr/bin/id -u)" -ne 0 ] || die "run the installer as your normal user, not with sudo"

install_succeeded=0
current_switched=0
launcher_changed=0
previous_current_present=0
previous_launcher_present=0
release_created=0
lock_owned=0
launcher_stage=""
new_current=""
rollback_link=""
uv_staging=""
ffmpeg_staging=""

test_mode=${DWELL_INSTALLER_TEST_MODE:-0}
case "$test_mode" in
    0|1) ;;
    *) die "DWELL_INSTALLER_TEST_MODE must be 0 or 1" ;;
esac

if [ "$test_mode" -eq 0 ]; then
    test_hook_names=$(
        /usr/bin/env | /usr/bin/sed -n 's/^\(DWELL_TEST_[A-Za-z0-9_]*\)=.*/\1/p'
    )
    [ -z "$test_hook_names" ] \
        || die "DWELL_TEST_* overrides require DWELL_INSTALLER_TEST_MODE=1"

    # Production downloads and privileged writes never resolve these commands
    # through a caller-controlled PATH.
    CURL=/usr/bin/curl
    SHASUM=/usr/bin/shasum
    SW_VERS=/usr/bin/sw_vers
    FILE_CMD=/usr/bin/file
    CODESIGN=/usr/bin/codesign
    TAR=/usr/bin/tar
    UNZIP=/usr/bin/unzip
    INSTALL_CMD=/usr/bin/install
    UNAME_CMD=/usr/bin/uname
else
    CURL=${DWELL_TEST_CURL:-curl}
    SHASUM=${DWELL_TEST_SHASUM:-shasum}
    SW_VERS=${DWELL_TEST_SW_VERS:-sw_vers}
    FILE_CMD=${DWELL_TEST_FILE:-file}
    CODESIGN=${DWELL_TEST_CODESIGN:-codesign}
    TAR=${DWELL_TEST_TAR:-tar}
    UNZIP=${DWELL_TEST_UNZIP:-unzip}
    INSTALL_CMD=${DWELL_TEST_INSTALL:-install}
    UNAME_CMD=${DWELL_TEST_UNAME:-uname}
fi

[ -n "${HOME:-}" ] || die "HOME is not set"
validate_path HOME "$HOME"

for command in "$CURL" "$SHASUM" "$FILE_CMD" "$CODESIGN" "$TAR" "$UNZIP" \
    "$INSTALL_CMD" "$UNAME_CMD" awk cat cp cut env grep sed wc tr chmod \
    mkdir mktemp mv ln readlink rm rmdir kill; do
    require_command "$command"
done

validate_platform

install_root=${DWELL_INSTALL_ROOT:-"$HOME/.local/share/dwell"}
bin_dir=${DWELL_BIN_DIR:-/usr/local/bin}
validate_path DWELL_INSTALL_ROOT "$install_root"
validate_path DWELL_BIN_DIR "$bin_dir"
[ "$install_root" != "$HOME" ] || die "DWELL_INSTALL_ROOT may not be HOME"
[ ! -L "$install_root" ] || die "DWELL_INSTALL_ROOT may not be a symlink"
[ ! -L "$bin_dir" ] || die "DWELL_BIN_DIR may not be a symlink: $bin_dir"
path_contains_directory "$bin_dir" \
    || die "DWELL_BIN_DIR is not present in PATH: $bin_dir"

launcher_target="$bin_dir/dwell"
validate_launcher_target "$launcher_target"
existing_dwell=$(command -v dwell 2>/dev/null || true)
if [ -n "$existing_dwell" ] && [ "$existing_dwell" != "$launcher_target" ]; then
    die "another dwell command would shadow this install: $existing_dwell; remove it first (for Homebrew: brew uninstall dwell), run hash -r, then rerun"
fi

use_sudo=0
if [ -d "$bin_dir" ] && [ -w "$bin_dir" ]; then
    :
elif [ -n "${DWELL_BIN_DIR:-}" ]; then
    die "DWELL_BIN_DIR is not a writable directory: $bin_dir"
elif [ "$test_mode" -eq 1 ]; then
    die "test installs require a writable DWELL_BIN_DIR"
else
    [ "$bin_dir" = "/usr/local/bin" ] \
        || die "privileged launcher installation is restricted to /usr/local/bin"
    require_command /usr/bin/sudo
    require_command /usr/bin/install
    require_command /bin/mkdir
    require_command /bin/mv
    require_command /bin/rm
    use_sudo=1
    /usr/bin/sudo -v
    /usr/bin/sudo /bin/mkdir -p "$bin_dir"
    [ -d "$bin_dir" ] && [ ! -L "$bin_dir" ] \
        || die "launcher directory is not a safe directory: $bin_dir"
fi

mkdir -p "$install_root"
install_lock="$install_root/.install-lock"
acquire_install_lock

temporary_root=$(mktemp -d "${TMPDIR:-/tmp}/dwell-install.XXXXXX")
releases_dir="$install_root/releases"
tools_root="$install_root/tools"
python_root="$install_root/python"
cache_root="$install_root/cache"
mkdir -p "$releases_dir" "$tools_root" "$python_root" "$cache_root"

release_base=${DWELL_RELEASE_BASE_URL:-}
if [ -z "$release_base" ]; then
    if [ -n "${DWELL_VERSION:-}" ]; then
        requested_version=${DWELL_VERSION#v}
        release_base="https://github.com/$REPOSITORY/releases/download/v$requested_version"
    else
        requested_version=""
        release_base="https://github.com/$REPOSITORY/releases/latest/download"
    fi
else
    requested_version=${DWELL_VERSION:-}
    requested_version=${requested_version#v}
fi
[ -z "$requested_version" ] || is_release_version "$requested_version" \
    || die "invalid DWELL_VERSION: ${DWELL_VERSION:-}"
release_base=${release_base%/}
case "$release_base" in
    https://*) ;;
    *) die "release downloads must use HTTPS: $release_base" ;;
esac

manifest="$temporary_root/SHA256SUMS"
download "$release_base/SHA256SUMS" "$manifest"

wheel_pattern='^dwell_ai-[0-9]+\.[0-9]+\.[0-9]+-py3-none-any\.whl$'
wheel_count=$(awk -v pattern="$wheel_pattern" \
    '$2 ~ pattern { count += 1 } END { print count + 0 }' "$manifest")
[ "$wheel_count" -eq 1 ] || die "SHA256SUMS must identify exactly one Dwell wheel"
wheel_name=$(awk -v pattern="$wheel_pattern" '$2 ~ pattern { print $2 }' "$manifest")
version=${wheel_name#dwell_ai-}
version=${version%-py3-none-any.whl}
[ -z "$requested_version" ] || [ "$requested_version" = "$version" ] \
    || die "release version $version does not match requested version $requested_version"

wheel_sha=$(checksum_from_manifest "$wheel_name" "$manifest")
requirements_sha=$(checksum_from_manifest "$REQUIREMENTS_NAME" "$manifest")
wheel="$temporary_root/$wheel_name"
requirements="$temporary_root/$REQUIREMENTS_NAME"
download "$release_base/$wheel_name" "$wheel"
download "$release_base/$REQUIREMENTS_NAME" "$requirements"
verify_sha256 "$wheel" "$wheel_sha" "$wheel_name"
verify_sha256 "$requirements" "$requirements_sha" "$REQUIREMENTS_NAME"

UV_ARCHIVE_URL=${DWELL_TEST_UV_ARCHIVE_URL:-$UV_ARCHIVE_URL_DEFAULT}
UV_ARCHIVE_SHA256=${DWELL_TEST_UV_ARCHIVE_SHA256:-$UV_ARCHIVE_SHA256_DEFAULT}
UV_BINARY_SHA256=${DWELL_TEST_UV_BINARY_SHA256:-$UV_BINARY_SHA256_DEFAULT}
FFMPEG_URL=${DWELL_TEST_FFMPEG_URL:-$FFMPEG_URL_DEFAULT}
FFMPEG_ARCHIVE_SHA256=${DWELL_TEST_FFMPEG_ARCHIVE_SHA256:-$FFMPEG_ARCHIVE_SHA256_DEFAULT}
FFMPEG_BINARY_SHA256=${DWELL_TEST_FFMPEG_BINARY_SHA256:-$FFMPEG_BINARY_SHA256_DEFAULT}
FFPROBE_URL=${DWELL_TEST_FFPROBE_URL:-$FFPROBE_URL_DEFAULT}
FFPROBE_ARCHIVE_SHA256=${DWELL_TEST_FFPROBE_ARCHIVE_SHA256:-$FFPROBE_ARCHIVE_SHA256_DEFAULT}
FFPROBE_BINARY_SHA256=${DWELL_TEST_FFPROBE_BINARY_SHA256:-$FFPROBE_BINARY_SHA256_DEFAULT}

# Prevent an ambient development configuration from changing the install.
sanitize_package_environment
export UV_CACHE_DIR="$cache_root"
export UV_PYTHON_INSTALL_DIR="$python_root"

prepare_uv
prepare_ffmpeg

current="$install_root/current"
if [ -e "$current" ] && [ ! -L "$current" ]; then
    die "managed current path is not a symlink: $current"
fi
if [ -L "$current" ]; then
    previous_current_target=$(readlink "$current")
    previous_current_present=1
fi

previous_launcher="$temporary_root/previous-launcher"
if [ -e "$launcher_target" ]; then
    "$INSTALL_CMD" -m 0755 "$launcher_target" "$previous_launcher"
    previous_launcher_present=1
fi

already_current=0
if [ -L "$current" ] && [ -x "$current/bin/dwell" ] \
    && [ -f "$current/.dwell-install-metadata" ]; then
    if grep -Fqx "wheel_sha256=$wheel_sha" "$current/.dwell-install-metadata" \
        && grep -Fqx "requirements_sha256=$requirements_sha" "$current/.dwell-install-metadata"; then
        already_current=1
    fi
fi

if [ "$already_current" -eq 0 ]; then
    release_prefix=$(printf '%s' "$wheel_sha" | cut -c 1-12)
    release_dir=$(mktemp -d "$releases_dir/$version-$release_prefix.XXXXXX")
    release_created=1

    "$UV_BIN" --quiet --no-config python install "$UV_PYTHON_VERSION" --no-bin \
        --install-dir "$python_root"
    "$UV_BIN" --quiet --no-config venv "$release_dir" --python "$UV_PYTHON_VERSION" \
        --managed-python
    "$UV_BIN" --quiet --no-config pip sync "$requirements" \
        --python "$release_dir/bin/python" \
        --managed-python \
        --require-hashes \
        --no-build \
        --strict
    "$UV_BIN" --quiet --no-config pip install "$wheel" \
        --python "$release_dir/bin/python" \
        --managed-python \
        --no-deps \
        --no-build

    [ "$("$release_dir/bin/dwell" --version)" = "Dwell $version" ] \
        || die "installed Dwell CLI failed its version smoke test"
    cat >"$release_dir/.dwell-install-metadata" <<EOF
version=$version
wheel_sha256=$wheel_sha
requirements_sha256=$requirements_sha
uv_version=$UV_VERSION
python_version=$UV_PYTHON_VERSION
ffmpeg_version=$FFMPEG_VERSION
EOF

else
    release_dir=$(readlink "$current")
fi

launcher="$temporary_root/dwell-launcher"
write_launcher "$launcher"
launcher_changed=1
install_launcher_atomically "$launcher" "$launcher_target" \
    || die "failed to install the Dwell launcher"

if [ "$already_current" -eq 0 ]; then
    new_current="$install_root/.current-$$"
    [ ! -e "$new_current" ] && [ ! -L "$new_current" ] \
        || die "temporary current link already exists: $new_current"
    ln -s "$release_dir" "$new_current"
    # Mark the switch before publishing it, so an interrupt can always restore
    # the previous target. Cleanup never removes a release that remains active.
    current_switched=1
    replace_symlink_atomically "$new_current" "$current"
fi

[ "$("$launcher_target" --version)" = "Dwell $version" ] \
    || die "installed launcher failed its version smoke test"

release_created=0
install_succeeded=1
say "Dwell $version installed successfully."
say "Command: $launcher_target"
say "Run: dwell --help"
