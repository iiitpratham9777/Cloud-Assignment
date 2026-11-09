# Portability changes: what changed and what was verified

This records the work that made the project portable (local mode, installation, configuration, checks, documentation).
Nothing in the original functionality was removed: the Kubernetes/S3 pipeline, scripts, report and credits are all still there.
The original upload is the first commit of the working history ("baseline").

## Changes

| Area | Change |
|---|---|
| Configuration | `service/settings.py`: environment variables + optional `.env`; paths resolved from the repository root, not the current directory. `.env.example` documents every variable. Blank values never shadow real settings. |
| Local mode | API loads a local model folder (`MODEL_DIR`, default `model_out/v1`) when `MODEL_URI` is not set; `/readyz` explains why it is not ready. `MODEL_URI` (S3) behaviour is unchanged. Model cache is a portable temp folder instead of `/tmp/model`. `X-Served-By` falls back to the host name (Windows has no `HOSTNAME`). |
| Checks | `service/checks.py`, `scripts/check_setup.py`: Python version, packages, datasets, weights, with fix instructions; used by `train.py`, the API and `dev.py serve`. |
| Install | `scripts/dev.py` (one cross-platform entry point: setup, check, serve, test, lint, train, smoke, audit); `requirements/` split by purpose; macOS-safe torch pin. `Makefile` gained local targets. |
| Data | `scripts/label_local.py` (teacher labelling without Kubernetes, same request/sharding as the worker), `scripts/make_smoke_dataset.py` (synthetic, testing only). |
| Weights | `scripts/fetch_model.py`: download a release archive, verify SHA-256, safe extraction. |
| Training | `training/train.py`: repo-root paths, friendly dataset error, `--train-file/--base-model/--output-dir/--loss-csv/--threads/--save`; smoke runs never touch the tracked `models/*/model.sha256`. `evaluate.py` works from any folder and defaults to `MODEL_DIR`. |
| Safety | hardened `.gitignore`; `scripts/audit_repo.py` secret/personal-path/weights scan. |
| Optional cluster | `scripts/up.sh` stops with a clear message if `REPO_SLUG` is still the placeholder; CI skips model evaluation on repositories without a `model-release` Release. |
| Docs | README (Part 1 local, Part 2 original cluster text), `CONTRIBUTING.md`, `ATTRIBUTION.md`, `LICENSING.md`, `docs/PUBLISHING.md`, `docs/LICENSE.mit.template`. |
| Tests | 7 new test files; docs-consistency tests; an integration test that runs the real training and inference code on a tiny model. |

## Verification record (sandbox run, Linux, Python 3.12.3)

Environment: PyTorch 2.4.1 CPU build installed from conda, all other packages installed from PyPI at the exact pinned versions.

| Check | Result |
|---|---|
| Original code, original tests (baseline) | 15 passed; ruff clean |
| Final code: `pytest` | see the run recorded in the hand-over message (all pass, 0 failures) |
| `dev.py setup --no-ml` on a clean copy without `.git` | created `.venv`, installed pinned packages, created `.env`, ran the checker |
| `dev.py serve` without weights | refused with instructions; `--allow-no-model` started the API: `/healthz` 200, `/readyz` 503 + detail, `/parse` 503, `/docs` 200 |
| Training code, real PyTorch, **tiny random model** | trained, saved, archived with checksum, fetched with verification, loaded, generated, served through the API (HTTP 200 for a training-set example) |
| `dev.py smoke --base-model <tiny>` then `dev.py serve` | model loaded (`/readyz` 200); `/parse` returned 500 `invalid_model_output` (3 epochs on a random tiny model: expected) |
| `evaluate.py model` run from another directory on the smoke model | ran; 0 % validity: the smoke model is meaningless, **not** a project result |
| Local labelling | tested only against a **fake** Ollama HTTP server |
| `up.sh` placeholder guard | tested with stubbed tools: guard triggers and exits non-zero |
| Publishing flow (`git init`, audit, commit, push) | rehearsed against a local bare repository (see below) |

## NOT verified

* Training or inference with the real **EleutherAI/pythia-70m** (Hugging Face was unreachable from the sandbox) and therefore **no
  real accuracy numbers exist** for this project from this work.
* The real Ollama teacher (`phi3.5`).
* `pip install -r requirements/ml.txt` with the `torch==2.4.1+cpu` pin from `download.pytorch.org` (that host was unreachable).
* Windows and macOS (the code uses `pathlib`/argument lists and is intended to work, but it was not run there).
* Docker image build, k3d/Kubernetes, SeaweedFS, the load tests and the GitHub Actions workflow (YAML parses; never executed).
* Pushing to real GitHub.

Please run these on your own machine and record the real outcomes in `STATUS.md`.
