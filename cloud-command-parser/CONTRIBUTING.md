# Contributing / customising this project

This guide is for anyone who downloads the project, wants to change it for their own purposes and, optionally, share the result.
Read [`README.md`](README.md) Part 1 first and make sure `python scripts/dev.py test` passes before you change anything.

> **Credits come first.** This project's original author is **Vivek Sagarwal**; this copy was prepared by **Pratham Gupta**.
> Keep those credits (README, `ATTRIBUTION.md`, `AI_USE.md`, `STATUS.md`, report) in any copy you share and add your own name
> next to them, never instead of them. Read [`LICENSING.md`](LICENSING.md) before publishing: no licence has been granted yet.

## 1. Set up a development copy

```bash
python scripts/dev.py setup        # Windows: py -3.11 scripts\dev.py setup
python scripts/dev.py test         # must be green before you start
python scripts/dev.py lint
```

Work on a branch: `git switch -c my-change`.

## 2. Map of the code

| Path | What it does | Typical changes |
|---|---|---|
| `service/app.py` | FastAPI endpoints (`/healthz`, `/readyz`, `/parse`) | add endpoints, change responses |
| `service/model_loader.py` | loads a model folder (or downloads from S3) and generates text | change decoding |
| `service/parsing.py` | prompt format shared by training, evaluation and serving | change the prompt (retrain afterwards!) |
| `service/validator.py` + `data/schema.json` | checks that model output is a valid frame | add intents or slots |
| `service/settings.py` | environment/`.env` handling and path resolution | add a setting |
| `service/checks.py` | pre-flight checks for datasets and weights | new required files |
| `training/train.py`, `training/config.yaml` | fine-tuning and hyper-parameters | change model, epochs, learning rate |
| `worker/` | teacher labelling for Kubernetes + `clean.py` (dataset cleaning rules) | change cleaning rules |
| `scripts/` | `dev.py` (entry point), `label_local.py`, `fetch_model.py`, `evaluate.py`, `audit_repo.py`, ... | new helpers |
| `prompts/teacher_prompt.txt` | prompt given to the teacher model | change the labelling instructions |
| `tests/` | unit tests; `test_integration_tiny.py` runs the real training code on a tiny model | add tests |
| `k8s/`, `cluster/`, `Dockerfile*`, `scripts/*.sh` | optional Kubernetes deployment | only if you use it |
| `requirements/` | dependency files (see `requirements/README.md`) | upgrade versions |

## 3. Common customisations

**Add a new intent** (for example `music_play`): add it to `data/schema.json` (`intents`), to `INTENTS` in `worker/clean.py`,
and to the intent list in `prompts/teacher_prompt.txt`. Add or adjust tests in `tests/`, regenerate/clean the training data
(README section 6), retrain, and evaluate. Existing models will not know the new intent until retrained.

**Change the training recipe:** edit `training/config.yaml` (or pass `--base-model`, `--train-file`, `--output-dir`,
`--threads`). If you change the base model, check that `service/model_loader.py` still loads it, and update
`library_versions` if you change library pins.

**Use your own commands:** replace/extend `data/commands_pool.jsonl` (`{"id": "...", "text": "..."}` per line, unique ids),
check `ATTRIBUTION.md` for the rules on the original data, re-run labelling and cleaning.

**Change dependencies:** edit the pin in *every* file that lists the package (`requirements/*.txt`,
`requirements-dev.txt`, `service/requirements.txt`, `training/config.yaml` for torch/transformers/tokenizers/numpy).
`python -m pytest tests/test_requirements.py` fails when they disagree. Stay within Python 3.10-3.12 unless you also
upgrade PyTorch.

## 4. Before you commit

```bash
python scripts/dev.py lint
python scripts/dev.py test
python scripts/dev.py audit      # secrets, personal paths, accidentally added weights
git status                        # read the list of files: is anything surprising there?
```

Rules:

* **Never commit** `.env`, `.secrets/`, passwords, API keys, tokens, `model_out/`, weights (`*.safetensors`, `*.bin`, `*.pt`,
  `*.tar.gz`) or absolute paths of your own computer. `.gitignore` covers the usual cases; `audit` double-checks.
* Use environment variables for configuration and add new ones to `.env.example` (with a harmless placeholder) and to the
  README table.
* Do not invent results. Numbers in `STATUS.md`, `report/` and `results/` must come from runs you did; label anything not run
  as pending.
* Keep new code portable: use `pathlib` / `service.settings.resolve_path`, never hard-code `/tmp`, `C:\` or `/home/...`, and
  call external programs with argument lists (no shell-specific syntax) unless the file is explicitly a bash script.
* If you used an AI tool, update `AI_USE.md` honestly.

## 5. Tests

* `python scripts/dev.py test` runs everything; extra arguments go to pytest, e.g.
  `python scripts/dev.py test tests/test_checks.py -k model`.
* Tests must not need internet access, Ollama, GPUs, Docker or Kubernetes. The integration test uses a tiny random model
  built locally; it verifies that the code runs, not model quality.
* A change to training, serving or data handling should come with a test.

## 6. Sharing your changes

* Publishing your own copy: [`docs/PUBLISHING.md`](docs/PUBLISHING.md).
* Contributing back to a repository you were given access to: push your branch and open a pull request; describe what you
  changed, how you tested it and which results (if any) you measured yourself.
* CI (`.github/workflows/ci.yml`) runs a secret scan, lint and tests. The model evaluation / quality-gate jobs are skipped on
  repositories that have no `model-release` GitHub Release; publish `model-v1.tar.gz` as a release asset to enable them.

## 7. Optional: Kubernetes workflow

Only needed if you change `k8s/`, `Dockerfile*` or the worker. See README Part 2. Needs Docker, k3d, kubectl, bash
(WSL 2 on Windows). These files were **not** exercised when the portability changes were made (see `docs/CHANGES.md`), so test
them yourself before relying on them.
