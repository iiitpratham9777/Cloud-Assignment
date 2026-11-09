# Cloud Command Parser: knowledge distillation and deployment on a local Kubernetes cloud

Student working copy: **Pratham Gupta** · Roll number: **230103034**

Original project author: **Vivek Sagarwal**. This is a locally customized study copy; retain this attribution when reusing or submitting project material. Add your own repository and release URLs only after creating and verifying them.

> **Status:** see `STATUS.md` for what is implemented, what was executed and what is still pending.
> Anything marked *PENDING* has not been run yet; no result in this repository is invented.

> **Licence note:** the original archive contained no `LICENSE` file, so no reuse rights have been granted yet.
> Read [`LICENSING.md`](LICENSING.md) and [`ATTRIBUTION.md`](ATTRIBUTION.md) before you publish or share a copy.

---

# Part 1 - Run it on your own computer (no Docker, no Kubernetes)

This part is all you need to try, modify and extend the project on a laptop.

## 1. What you need

* **Python 3.10, 3.11 or 3.12** (3.11 recommended). Python 3.13 and newer are **not** supported by the pinned PyTorch.
  Check with `python --version` (Windows: `py --list`).
* **Git** (only needed to publish your own version) and an internet connection for the first install.
* A few GB of free disk space (PyTorch is large) plus room for the model files.
* Optional: [Ollama](https://ollama.com) - only if you want to regenerate the training labels yourself.

Docker, Kubernetes, k3d, kubectl and `make` are **not** needed for anything in Part 1.

## 2. Install (one command)

Open a terminal in the project folder (the folder that contains this README).

| System | Command |
|---|---|
| Linux / macOS | `python3 scripts/dev.py setup` |
| Windows (PowerShell or cmd) | `py -3.11 scripts\dev.py setup` |

This creates a private virtual environment in `.venv/`, installs the pinned dependencies from `requirements/`,
copies `.env.example` to `.env` and prints a status report. You never have to "activate" the environment:
every `scripts/dev.py` command uses it automatically.

* Only want the API and the tests, without PyTorch? Add `--no-ml` (small and quick).
* Need to start over? Add `--recreate`.
* Prefer doing it by hand?
  `python -m venv .venv`, activate it (`source .venv/bin/activate` on Linux/macOS,
  `.venv\Scripts\Activate.ps1` on Windows), then `pip install -r requirements/local.txt`.

On macOS the install uses PyTorch's normal PyPI wheel; on Linux and Windows it uses the smaller CPU-only build
(see `requirements/README.md`).

## 3. Check your setup

```bash
python scripts/dev.py check            # general report
python scripts/dev.py check serve      # is everything there to run the API?
python scripts/dev.py check train      # is everything there to train?
```

The checker lists missing Python packages, missing datasets and missing model weights, and says how to fix each one.
`[WARN]` means "optional for what you are doing", `[FAIL]` means "this command cannot work yet".

## 4. Run the unit tests

```bash
python scripts/dev.py test
python scripts/dev.py lint
```

Tests that need PyTorch are skipped automatically when it is not installed.

## 5. Model weights: get them, or train them

The API needs a trained model folder (default `model_out/v1/`, containing `config.json`, `model.safetensors` and tokenizer
files). Weights are **never committed to Git** (they are in `.gitignore`). Choose one:

**A. Download pretrained weights** (no training). Works if someone published `model-v1.tar.gz` as a GitHub Release asset:

```bash
python scripts/fetch_model.py --repo OWNER/REPO --tag model-release --version v1
```

The download is checked against a SHA-256 checksum before it is unpacked (from `models/v1/model.sha256`, or pass
`--sha256 HEX`). Only download weights from people you trust. Use any other folder with `MODEL_DIR=/path/to/model` in `.env`.

**B. Train it yourself** - see sections 6 and 7.

## 6. Training data

Training needs `data/train.jsonl`: one JSON object per line with `id`, `text` and `teacher_raw` (the teacher model's raw
JSON answer). **This file is not in the repository.** Three ways to get it:

1. **Copy an existing one.** If you or a colleague already produced a cleaned `data/train.jsonl`, put it at that path.
2. **Generate it with the teacher model (Ollama).** This reproduces the original method and can take hours on a CPU laptop:
   ```bash
   ollama pull phi3.5                      # once; Ollama must be running
   python scripts/label_local.py           # labels data/commands_pool.jsonl -> results/teacher/shard-0.jsonl
   python -m worker.clean --local results/teacher --out data     # validate + deduplicate -> data/train.jsonl
   ```
   Try `python scripts/label_local.py --limit 50` first. The script can be stopped and restarted (it resumes).
   Two laptops can split the work: `--shard-index 0 --shard-count 2` on one and `--shard-index 1 --shard-count 2`
   on the other, then copy both `shard-*.jsonl` files into one folder before cleaning.
3. **Synthetic smoke dataset (testing only).** `python scripts/make_smoke_dataset.py` writes a tiny template-based
   dataset to `data/smoke_train.jsonl`. It proves the training code runs; a model trained on it is **not** useful and its
   numbers must not be reported as results.

The 3,000 raw commands live in `data/commands_pool.jsonl` (see `ATTRIBUTION.md` for their origin) and the 100 labelled
evaluation examples in `data/validation.jsonl`.

## 7. Train a model

```bash
python scripts/dev.py train --subset 50          # quick check, writes training/loss_subset.csv, saves nothing
python scripts/dev.py train --version v1         # full run -> model_out/v1/ + model_out/model-v1.tar.gz
```

* Settings (learning rate, epochs, threads, ...) are in `training/config.yaml`. Lower `threads` on a small laptop.
* The first run downloads the base model `EleutherAI/pythia-70m` from Hugging Face, so it needs internet access.
  Use `--base-model /path/to/folder` to start from a local copy.
* A full run also writes `models/v1/model.sha256` (the checksum of the release archive).
* Tiny end-to-end check on synthetic data, saved to `model_out/smoke/v1`:
  `python scripts/dev.py smoke` (add `--base-model DIR` to avoid the download).
* Evaluate on the 100 validation examples: `python scripts/evaluate.py model --out results/student_predictions.jsonl`
  (uses `MODEL_DIR` unless you pass `--model`). Only numbers you produce yourself belong in your report.

## 8. Run the API locally

```bash
python scripts/dev.py serve
```

Open <http://127.0.0.1:8000/docs> for an interactive page, or from a second terminal:

```bash
curl -s -X POST http://127.0.0.1:8000/parse -H "Content-Type: application/json" \
     -d '{"text":"what is the weather in london"}'
```

PowerShell: `Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/parse -ContentType "application/json" -Body '{"text":"what is the weather in london"}'`

| Endpoint | Meaning |
|---|---|
| `GET /healthz` | process is alive (always 200) |
| `GET /readyz` | 200 when the model is loaded, otherwise 503 (with a `detail` explaining why, e.g. missing weights) |
| `POST /parse` | body `{"text": "..."}` -> `{"intent","date","time","place"}`; 503 if the model is not ready; 500 `{"error":"invalid_model_output"}` if the model produced invalid JSON |

`serve` refuses to start without weights and prints how to get them (`--allow-no-model` starts it anyway).
It listens on `127.0.0.1` (this computer only); change `HOST`/`PORT` in `.env` or use `--host`/`--port`.
There is no authentication, so do not expose it to the internet as is.

## 9. Configuration (`.env`)

All settings are environment variables. `.env.example` documents every one; `.env` (git-ignored) is your private copy.
Real environment variables always override `.env`. Relative paths are relative to the project folder.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `model_out/v1` | local model folder used by the API and `evaluate.py` |
| `MODEL_VERSION` | `v1` | returned in the `X-Model-Version` header |
| `MAX_NEW_TOKENS`, `TORCH_THREADS` | `64`, `2` | generation length, CPU threads while serving |
| `HOST`, `PORT` | `127.0.0.1`, `8000` | where `dev.py serve` listens |
| `OLLAMA_URL`, `TEACHER_MODEL` | `http://localhost:11434`, `phi3.5` | teacher for `label_local.py` |
| `MODEL_URI`, `S3_ENDPOINT`, `AWS_*` | empty | only for the optional Kubernetes/S3 mode (Part 2) |
| `MODEL_CACHE_DIR` | system temp folder | where a model downloaded in S3 mode is cached |
| `FAULT_INJECT` | `0` | `1` forces an invalid reply to test the error path |

**Never put real passwords, tokens or keys in `.env.example`, and never commit `.env`.**

## 10. Customise and publish your own version

* How the code is organised and how to change it: [`CONTRIBUTING.md`](CONTRIBUTING.md).
* Creating your own GitHub repository and pushing (keeping the credits intact): [`docs/PUBLISHING.md`](docs/PUBLISHING.md).
* Licence and credits: [`LICENSING.md`](LICENSING.md), [`ATTRIBUTION.md`](ATTRIBUTION.md).
* Before every push run `python scripts/dev.py audit` - it looks for committed secrets, personal paths and model files.

## 11. Troubleshooting

| Problem | Fix |
|---|---|
| `Python ... is not supported` | install Python 3.10-3.12 and run setup with that interpreter (`py -3.11 ...`, `python3.11 ...`) |
| Linux: venv creation fails | `sudo apt install python3-venv` |
| `Model directory not found` | section 5 (download or train) or set `MODEL_DIR` |
| `Training dataset not found` | section 6 |
| `cannot reach Ollama` | install/start Ollama (`ollama serve`) and `ollama pull phi3.5` |
| Port already in use | `python scripts/dev.py serve --port 8001` |
| `/parse` returns 500 `invalid_model_output` | the model produced text that is not valid JSON (normal for an undertrained model) |
| Out of memory while training | lower `batch_size` in `training/config.yaml` |

---

# Part 2 - Original Docker / Kubernetes pipeline (advanced, optional)

Everything below is the original project documentation, kept unchanged apart from headings. It needs Docker, k3d, kubectl,
`make` and bash (on Windows use WSL 2). You can ignore all of it for local use.

## What it does
A small voice-assistant style parser: one English command in, one JSON object out
(`intent`, `date`, `time`, `place`). The teacher (Phi-3.5-mini via Ollama, temperature 0, seed 42)
labels 3,000 SLURP commands in parallel workers; Pythia-70M is fine-tuned on those replies
(sequence-level distillation, CPU only) and served by a FastAPI container on k3d/Kubernetes.

## Architecture
```
commands_pool.jsonl --> SeaweedFS (datasets) --> 2 worker Pods (pipeline ns) --> Ollama phi3.5 (laptop)
                                   shard-0/1.jsonl --> worker/clean.py --> datasets/v1/train.jsonl (+sha256)
train.jsonl --> training/train.py (CPU) --> SeaweedFS (models/v1, models/v2) + GitHub Release assets
client --> Traefik Ingress :8080 --> Service --> Deployment (2 Pods, HPA 1-2) --> model from SeaweedFS at start-up
```
Namespaces: `data` (SeaweedFS), `pipeline` (labelling workers), `app` (service).

## Technologies
k3d/k3s, Kubernetes, SeaweedFS (S3 API), Ollama + phi3.5, PyTorch (CPU), Hugging Face Transformers,
FastAPI/uvicorn, Docker, Locust, GitHub Actions, gitleaks, ruff, pytest.

## Repository structure
`cluster/` k3d config · `k8s/` manifests · `worker/` sharded labelling + `clean.py` · `training/` config, trainer,
loss.csv · `service/` API · `tests/` · `scripts/` helpers · `network/` subnets + NACL · `data/`, `tools/` (provided kit,
unchanged) · `prompts/` teacher prompt · `results/` evidence · `models/vN/model.sha256` · `report/` · `reflection/`.

## Prerequisites for the Kubernetes workflow
Docker, k3d ≥ 5, kubectl, Python 3.11, make, curl, [Ollama](https://ollama.com) with `ollama pull phi3.5`
(only for re-labelling), `pip install locust` (load test). Roughly 8 GB RAM free.

## Cluster quick start (advanced, optional)
```bash
git clone <YOUR_REPOSITORY_URL> cloud-command-parser && cd cloud-command-parser
make up        # cluster, SeaweedFS, images, buckets, models from the Release, deployment
curl -s -X POST localhost:8080/parse -H 'Content-Type: application/json' -d '{"text":"what is the weather in london"}'
```
`make up` does **not** repeat the multi-hour teacher labelling; it uses `data/train.jsonl` and the Release models. Before deployment, configure `REPO_SLUG` and `RELEASE_TAG` in `scripts/up.sh` for a repository/release that you own and have verified.
Override the Release location with `REPO_SLUG=<user>/<repo> RELEASE_TAG=model-release make up`.

## Task 1: teacher labelling (run once, takes hours)
```bash
OLLAMA_HOST=0.0.0.0 ollama serve &            # laptop; Pods reach it at host.k3d.internal:11434
ollama pull phi3.5
kubectl apply -f k8s/worker-job.yaml           # 2 workers, SHARD_INDEX 0 and 1, namespace pipeline
kubectl -n pipeline logs -f -l job-name=teacher-labelling --prefix
kubectl -n data port-forward svc/seaweedfs 8333:8333 &  &&  source .secrets/s3.env
python -m worker.clean                          # -> datasets/v1/train.jsonl, train.sha256, data/clean_stats.json
python scripts/storage.py get datasets v1/train.jsonl data/train.jsonl
```
Teacher settings: `temperature 0`, `seed 42`, JSON mode (`format: json`). Prompt: `prompts/teacher_prompt.txt`.

## Task 2: training
```bash
pip install -r service/requirements.txt pyyaml==6.0.2
make train SUBSET=50          # smoke test (writes training/loss_subset.csv)
make train VERSION=v1         # full run, reads only training/config.yaml; writes training/loss.csv, model_out/v1/
make train VERSION=v2
python scripts/storage.py put-dir model_out/v1 models models/v1     # object storage
gh release create model-release model_out/model-v1.tar.gz model_out/model-v2.tar.gz   # GitHub Release assets
```
`models/vN/model.sha256` = SHA-256 of the Release archive `model-vN.tar.gz` (deterministic tar.gz).
Evaluation of teacher / untrained Pythia / student:
```bash
python scripts/evaluate.py teacher --out results/teacher_predictions.jsonl     # needs Ollama
python scripts/evaluate.py model --model EleutherAI/pythia-70m --out results/baseline_predictions.jsonl
python scripts/evaluate.py model --model model_out/v1 --out results/student_predictions.jsonl
```

## Task 3: container
```bash
docker build -t command-parser:1.0 .                 # one command, pinned dependencies, no weights
docker images command-parser:1.0                     # size must be <= 2 GB
kubectl -n data port-forward --address 0.0.0.0 svc/seaweedfs 8333:8333 &
bash scripts/container_limits_test.sh                # 50 sequential requests at 1 CPU / 1 GiB, peak memory
```
Environment variables: `MODEL_URI`, `MODEL_VERSION`, `S3_ENDPOINT`, `MAX_NEW_TOKENS`, `FAULT_INJECT` (=1 forces `not json`).
Credentials come only from the Kubernetes Secret `s3-credentials` (generated at deploy time, never committed).

## Task 4 and 7: Kubernetes experiments and load test
```bash
bash scripts/loadtest.sh 2m results/loadtest/loadtest        # standard Locust command
kubectl -n app set env deployment/command-parser MODEL_VERSION=v2 MODEL_URI=s3://models/v2/   # rolling update (run with a longer load test)
kubectl -n app rollout undo deployment/command-parser        # rollback
kubectl -n app delete pod <one-pod>                           # recovery
bash scripts/coldstart.sh                                     # 3 cold starts + median
python scripts/cost.py --replicas 2 --stats results/loadtest/loadtest_stats.csv
```
For the load-balancing run, scale to exactly two Ready replicas first (`kubectl -n app delete hpa command-parser`
or `kubectl -n app scale deployment command-parser --replicas=2`, then re-apply `k8s/hpa.yaml`).

## Task 5: network
`network/subnets.md`, `network/nacl.json`. Check: `python tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check` (6/6).

## Tests and CI
`make lint`, `make test`. `.github/workflows/ci.yml`: gitleaks → lint → unit tests → model evaluation (Release asset) → quality gate
(`score.py --min-validity 95`). Run the workflow manually with *use_untrained_baseline = true* to show the gate failing.

## Shutdown
`make down` (deletes the k3d cluster). `rm -rf .secrets model_out` removes local credentials and weights.

## Object storage
SeaweedFS S3 API, `data` namespace, port 8333. Buckets `datasets` (`commands_pool.jsonl`, `teacher/shard-N.jsonl`,
`v1/train.jsonl`, `v1/train.sha256`) and `models` (`v1/…`, `v2/…`).

## Network configuration
10.20.0.0/16 → app 10.20.0.0/22, public 10.20.4.0/24, data 10.20.5.0/25, management 10.20.5.128/27. NACL for the public subnet in `network/nacl.json`.

## Results and limitations
Measured results are in `results/` and in `report/report.pdf` (see `STATUS.md` for which are still pending).
Limitations: single-node k3d (no multi-machine fault tolerance), 70M-parameter student with a CPU-only budget, labels
are the teacher's (errors are inherited), the HPA cannot show real scale-out capacity on one laptop.
