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
                 ├── text: future adapter
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
interface allows a future in-process or resident engine to implement real load/unload behavior.

## Installation with Homebrew

The repository is private, so first make sure your GitHub account has access and your SSH key can
authenticate to GitHub. Then install Dwell on an Apple Silicon Mac without cloning the repository
manually:

```bash
brew tap oktykrk/dwell git@github.com:oktykrk/dwell.git
brew install dwell
dwell setup
dwell models install ltx-2.5-bf16
dwell start
dwell status
```

`brew install` installs the Dwell CLI and its system dependencies in Homebrew's managed prefix.
`dwell setup` prepares the per-user runtime under `~/.dwell`; it does not download model weights,
start the server, or edit `.zshrc` or any other shell configuration. Model weights are downloaded
only by the explicit `dwell models install …` command.

### Updating a Homebrew installation

```bash
brew update
brew upgrade dwell
dwell setup --upgrade
dwell doctor
```

The setup upgrade updates the pinned runtime while preserving models, generated outputs, and job
data.

### Uninstalling Homebrew Dwell

```bash
dwell stop
brew uninstall dwell
```

`brew uninstall` removes the packaged CLI but does not delete user data. In particular,
`~/.dwell/models`, outputs, state, and logs are preserved. Automatic data deletion is intentionally
outside the MVP.

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

For development, `uv run dwell …` uses the project environment. A Homebrew installation links
`dwell` into Homebrew's normal `bin` directory; `~/.dwell/bin` and shell startup-file edits are not
required.

## Directory layout

```text
~/.dwell/
├── models/
│   └── huggingface/       shared HF_HOME (Hub and Xet caches are preserved)
├── runtimes/
│   └── ltx-2-mlx/         external runtime; remains its own Git repository
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

## Local API

The API listens only at `http://127.0.0.1:8188`.

```http
GET    /health
GET    /v1/status
GET    /v1/models
POST   /v1/videos
GET    /v1/jobs/{job_id}
DELETE /v1/jobs/{job_id}
```

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
project, package, and Homebrew validation without changing the remote repository.

For a new version, a publishing run captures the selected `main` commit as the immutable release
source, validates the matching `v<version>` tag and `Formula/dwell.rb` update, and then creates the
GitHub Release.
Commits that reach `main` while validation is running are preserved: the tag remains on the
captured source commit and the Formula update is added to the latest compatible `main`, with both
refs pushed atomically. A concurrent Formula change is never overwritten. Normal CI continues to
validate development commits and pull requests, but it does not publish them. If GitHub Release
creation is interrupted after the refs land, run the same version again from the new `main`; the
workflow verifies the existing refs and resumes the incomplete release.
