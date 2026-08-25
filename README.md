# Dwell

Dwell is a personal, localhost-only AI inference gateway for Apple Silicon. It gives local
applications one stable CLI and HTTP API while specialized runtimes continue to do the actual
inference.

Dwell is not a machine-learning framework. It is a model registry, lifecycle manager, inference
router, persistent job queue, runtime-adapter layer, local API, and CLI.

The central safety invariant is:

> Inference never downloads models. Only an explicit `dwell models install …` command may use the
> network for weights.

Startup, model inspection, API requests, runtime checks, and tests resolve model data from the
local Hugging Face cache only. A missing model produces `model_not_installed` before a job is
queued.

## Architecture

```text
Local applications
        │
        ▼
Dwell — 127.0.0.1:8188
        │
        ├── model registry and lifecycle
        ├── persistent single-worker job queue
        └── modality-specific engine interfaces
                 │
                 ├── video: ltx-2-mlx subprocess adapter
                 ├── text: daemon-owned mlx-lm sidecar adapter
                 ├── image: future adapter
                 ├── audio: future adapter
                 └── embeddings: future adapter
```

Applications know model IDs, request schemas, job IDs, and output paths. They do not need to know
about MLX virtual environments, Hugging Face snapshot paths, or runtime-specific commands.

## Native Apple Silicon strategy

Dwell launches inference directly on macOS. The LTX adapter runs the native MLX runtime in an
isolated subprocess, which gives cancellation and failure isolation without Docker. Heavy jobs use
one worker by default because CPU and GPU share unified memory.

The current LTX adapter is on-demand only. It does **not** keep a model resident between requests,
so `models load` reports readiness rather than pretending the model is loaded. `models unload`
truthfully reports that there is no persistent runtime state to release. The runtime capability
interface also supports the resident MLX-LM text engine. Dwell starts that sidecar lazily on the
first chat request, keeps the model available for later requests, and owns its shutdown.

## Installation

Install Dwell on an Apple Silicon Mac running macOS 14 Sonoma or newer with one command:

```bash
curl --proto '=https' --tlsv1.2 -fsSL \
  https://github.com/oktykrk/dwell/releases/latest/download/install.sh | sh
```

Then use the CLI directly:

```bash
dwell --version
dwell setup
dwell models install ltx-2.5-bf16
dwell start
dwell status
```

The installer downloads only checksummed release artifacts and prebuilt wheels. It privately
manages its pinned `uv`, Python, FFmpeg, and FFprobe versions under `~/.local/share/dwell`; the user
does not need to install or learn any of those tools. It never invokes a compiler and does not
require Homebrew. A small `dwell` launcher is installed into `/usr/local/bin`; if that directory is
not writable, the installer asks for `sudo` only while installing that launcher. Application data
and model weights remain separate under `~/.dwell`.

If `command -v dwell` already points to an older Homebrew or source-checkout installation, remove
that command first and run `hash -r`. For a Homebrew installation, use `brew uninstall dwell`.
The installer deliberately refuses to leave a new launcher hidden behind an older one, and this
migration does not remove anything under `~/.dwell`.

`dwell setup` prepares the pinned per-user LTX video runtime under `~/.dwell`. It does not download
model weights or start the server. Model weights are downloaded only by the explicit
`dwell models install …` command.

### Updating

```bash
curl --proto '=https' --tlsv1.2 -fsSL \
  https://github.com/oktykrk/dwell/releases/latest/download/install.sh | sh
dwell setup --upgrade
dwell doctor
```

The installer keeps the previous application version until the new version has passed its smoke
test, then switches atomically. The setup upgrade updates the pinned video runtime while preserving
models, generated outputs, and job data.

### Uninstalling the application

Stop Dwell, then remove the installer-managed launcher and application runtime:

```bash
dwell stop
sudo rm -f /usr/local/bin/dwell
rm -rf "$HOME/.local/share/dwell"
```

This preserves configuration, models, generated outputs, jobs, and logs under `~/.dwell`.

> **Warning:** To remove all Dwell data manually, run `rm -rf "$HOME/.dwell"` only after checking
> that the path is correct and retaining anything you need. This permanently deletes downloaded
> models, generated outputs, job state, configuration, runtimes, and logs and cannot be undone.

## Installation for development

Python 3.11 and [`uv`](https://docs.astral.sh/uv/) are expected.

```bash
uv sync
uv run dwell --help
```

The operational home defaults to `~/.dwell`. Source code may live elsewhere; caches, runtimes,
state, logs, temporary artifacts, and outputs remain under the operational home.

For development, `uv run dwell …` uses the project environment. The end-user installer and its
managed Python environment are independent from a source checkout.

## Directory layout

```text
~/.dwell/
├── models/
│   └── huggingface/       shared HF_HOME (Hub and Xet caches are preserved)
├── runtimes/
│   └── ltx-2-mlx/         pinned, checksum-verified external runtime
├── services/
│   └── api/               pointer/deployment location for the Dwell service
├── outputs/
│   ├── video/
│   ├── image/
│   ├── audio/
│   └── text/
├── state/                 PID metadata and jobs.sqlite
├── logs/                  dwell.log
├── tmp/
│   └── ltx-tests/
├── config/                editable central model registry
└── bin/                    reserved for local helpers; not required on PATH
```

Model caches, runtime repositories, generated media, logs, and state are deliberately excluded
from the Dwell source repository.

## Configuration

Defaults:

```text
DWELL_HOME=$HOME/.dwell
DWELL_HOST=127.0.0.1
DWELL_PORT=8188
DWELL_MLX_LM_PORT=8189
```

For safety, values other than `127.0.0.1` are rejected for `DWELL_HOST`. Resolved paths are shown
with:

```bash
dwell config show
```

For child processes, Dwell resolves the shared cache environment internally as:

```text
HF_HOME=$HOME/.dwell/models/huggingface
HF_HUB_CACHE=$HOME/.dwell/models/huggingface/hub
HF_XET_CACHE=$HOME/.dwell/models/huggingface/xet
```

These variables do not need to be exported in `.zshrc`. If they are unset, `dwell doctor` reports
the Dwell-managed values as healthy; an explicitly conflicting value produces a warning.

## CLI

Service management:

```bash
dwell start
dwell stop
dwell restart
dwell status
dwell logs
dwell logs --follow
dwell setup
dwell setup --check
dwell setup --repair
dwell setup --upgrade
dwell doctor
```

Model lifecycle:

```bash
dwell models list
dwell models ls
dwell models info <model>
dwell models install <model> --dry-run
dwell models install <model> [--yes]
dwell models remove <model> [--yes]
dwell models load <model>
dwell models unload <model>
dwell models unload --all
```

Jobs and outputs:

```bash
dwell jobs list
dwell jobs ls
dwell jobs show <job-id>
dwell jobs cancel <job-id>
dwell jobs clear
dwell outputs list
```

A safe first session is:

```bash
dwell setup
dwell setup --check
dwell doctor
dwell models list
dwell models install ltx-2.5-bf16 --dry-run
dwell models install qwen3-coder-30b-a3b-4bit --dry-run
dwell start
dwell status
dwell stop
```

`--dry-run` does not contact Hugging Face. A real install displays the source and approximate size,
available disk and unified memory, license and acceptable-use links, and the exact required files.
It asks for confirmation separately from runtime setup, uses the shared cache, resumes supported
partial downloads, and verifies required files before considering the model installed.

## Models and runtimes are separate

A runtime is executable software with capabilities. A model is a family/version/profile plus a
weight source. For example:

```text
runtime: ltx-2-mlx
model family/version: LTX 2.5
profile: bf16 or a separately sourced quantized profile
```

The registry distinguishes `registered`, `available`, `installed`, and `loaded`. The lifecycle
states are `not_installed`, `installed`, `loading`, `loaded`, `unloading`, and `error`.

The bundled registry records the locally established `mlx-community/ltx-2.5-mlx` bf16 repository.
It also records `ltx-2.5-q8` as an unconfigured profile: Dwell has not verified a compatible q8
repository, immutable revision, required-file set, and runtime behavior as one installable unit.
Dwell will not invent or automatically enable a source. Set and test an explicit registry override
in `~/.dwell/config/models.json` before attempting a real q8 installation.

The registry also pins `qwen3-coder-30b-a3b-4bit` to the verified
`mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit` snapshot. Install it explicitly with
`dwell models install`; after that, the Dwell daemon can serve it through the pinned `mlx-lm`
runtime and the OpenAI-compatible chat API.

## Local API

The API listens only at `http://127.0.0.1:8188`.

```http
GET    /health
GET    /v1/status
GET    /v1/models
POST   /v1/models/{model_id}/load
DELETE /v1/models/{model_id}/load
DELETE /v1/models
POST   /v1/videos
GET    /v1/jobs/{job_id}
DELETE /v1/jobs/{job_id}
GET    /openai/v1/models
POST   /openai/v1/chat/completions
```

### Local coding with Qwen and OpenCode

Install the weights once, then start the Dwell daemon:

```bash
dwell models install qwen3-coder-30b-a3b-4bit --yes
dwell start
```

No separate `mlx_lm.server` process or model-load command is required. The first chat request
starts a daemon-owned MLX-LM sidecar lazily; `dwell stop` shuts it down. Non-streaming requests use
the registered Dwell model ID:

```bash
curl -sS http://127.0.0.1:8188/openai/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-coder-30b-a3b-4bit",
    "messages": [
      {"role": "user", "content": "Write a Python binary-search function with tests."}
    ],
    "max_tokens": 512,
    "temperature": 0,
    "stream": false
  }'
```

For an SSE stream, set `stream` to `true` and keep curl's output unbuffered:

```bash
curl -N -sS http://127.0.0.1:8188/openai/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'accept: text/event-stream' \
  -d '{
    "model": "qwen3-coder-30b-a3b-4bit",
    "messages": [
      {"role": "user", "content": "Explain the failing test and propose a minimal patch."}
    ],
    "max_tokens": 512,
    "temperature": 0,
    "stream": true
  }'
```

OpenCode can use the same endpoint as a global OpenAI-compatible provider. Create
`~/.config/opencode/opencode.json`, or merge the `dwell` provider below into the existing file:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "dwell": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Dwell Local",
      "options": {
        "baseURL": "http://127.0.0.1:8188/openai/v1"
      },
      "models": {
        "qwen3-coder-30b-a3b-4bit": {
          "name": "Qwen3 Coder 30B A3B 4-bit",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  }
}
```

In Conductor, open **Settings → Harnesses → OpenCode**, make the provider/model visible, and
select `dwell/qwen3-coder-30b-a3b-4bit` in a new OpenCode chat. Use a **local** Conductor workspace:
a cloud workspace cannot reach the Mac's `127.0.0.1`. The model cache, daemon, fixed port, and Apple
Silicon GPU are shared across local workspaces, so run one Dwell daemon and keep only one text
generation active at a time.

Inference is offline-only: startup and chat requests resolve the exact installed snapshot and
never download weights. Only the explicit `dwell models install` command may contact Hugging Face.
The current compatibility surface targets OpenCode through OpenAI Chat Completions. OpenAI
Responses and Anthropic Messages are not implemented, so Codex and Claude Code cannot use this
endpoint as their model provider; Cursor is not supported for this local-provider path either.

Example video request:

```bash
curl -sS http://127.0.0.1:8188/v1/videos \
  -H 'content-type: application/json' \
  -d '{
    "model": "ltx-2.5-q8",
    "prompt": "A cinematic detective walking through rainy Istanbul at night",
    "width": 576,
    "height": 1024,
    "frames": 121,
    "fps": 24,
    "seed": 42
  }'
```

If that model is absent, the request performs no install and returns:

```json
{
  "error": {
    "code": "model_not_installed",
    "message": "Model 'ltx-2.5-q8' is not installed locally.",
    "details": null
  }
}
```

Successful submissions return a queued job ID. Poll `/v1/jobs/{job_id}` for one of `queued`,
`running`, `completed`, `failed`, or `cancelled`. Progress stays `null` when the underlying runtime
does not provide a trustworthy value. Video output is written once to
`~/.dwell/outputs/video/<job-id>.mp4`.

Set this in future applications:

```bash
export DWELL_URL=http://127.0.0.1:8188
```

## Job persistence and cancellation

Job history is stored in SQLite without an ORM. Exactly one GPU-heavy worker executes jobs. Queued
jobs survive a server restart; work found in `running` state after an interruption is marked failed
rather than incorrectly reported as active. Cancelling a running LTX job terminates its child
process, waits briefly, and kills it only if graceful termination fails.

`dwell jobs clear` removes terminal history only. It does not cancel or remove queued/running work.
Generated media is never automatically deleted.

## Adding another engine

1. Add a concrete modality-specific engine (`VideoEngine`, `TextEngine`, `ImageEngine`, and so on).
2. Implement the runtime lifecycle checks and declare honest capabilities such as persistent
   loading, progress reporting, cancellation, streaming, and structured output.
3. Add verified model metadata to the central registry. Do not place model-specific logic in HTTP
   routes.
4. Make installation explicit and keep local resolution separate from the networked installer.
5. Add offline tests for missing and partial weights before adding an inference test.

Future text adapters can add JSON Schema response formats without forcing video/image/audio engines
into a misleading universal `generate()` method.

## Development and offline verification

Run the test suite with model hubs disabled:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1 uv run pytest
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1 uv run ruff check .
```

Tests cover configuration, registry state, CLI parsing, API startup, errors, SQLite jobs, dry-run
installation, and process management without model weights. Do not run a real
`dwell models install` or LTX generation as part of routine tests.

## Maintainer release

Releases never run automatically on a commit or tag push. In GitHub Actions, open the **Release**
workflow, choose **Run workflow** from `main`, enter the version already recorded in
`pyproject.toml`, and select **publish**. Leaving **publish** unselected performs the complete
project, package, installer, and checksum validation without changing the remote repository.

A publishing run captures the selected `main` commit as the immutable release source, creates its
annotated `v<version>` tag without force, and publishes the wheel, source distribution, locked
macOS requirements, installer, and `SHA256SUMS`. The release workflow uses an Ubuntu runner and
prebuilt packages; it does not build a Homebrew formula or bottle.

Normal CI continues to validate development commits and pull requests, but it never publishes
them. If a publishing run is interrupted, dispatch the same version again. The workflow verifies
the existing tag and asset digests, repairs an incomplete draft, and never overwrites an already
published release.

## Managed FFmpeg

The installer downloads the pinned native Apple Silicon FFmpeg and FFprobe executables directly
from [Martin Riedl's signed macOS builds](https://ffmpeg.martin-riedl.de/), verifies their published
SHA-256 values and the publisher's Apple Developer ID, and keeps them private to Dwell. Dwell's
shell installer and Python package do not require the Dwell maintainer to hold a Developer ID.
Those executables are separate GPLv3 programs; their reproducible build source is published in the
[FFmpeg build-script repository](https://git.martin-riedl.de/ffmpeg/build-script). Dwell invokes
them as subprocesses and does not redistribute them in its own MIT-licensed release assets.
