# Requirements

| File | Contents | Needed for |
|---|---|---|
| `api.txt` | FastAPI, pydantic, starlette, uvicorn | running the API |
| `ml.txt` | CPU PyTorch, transformers, tokenizers, safetensors, numpy, pyyaml | training **and** inference |
| `s3.txt` | boto3 | optional Kubernetes / S3 workflow only |
| `local.txt` | api + ml + `../requirements-dev.txt` (pytest, httpx, ruff) | everyday laptop use (`dev.py setup`) |
| `all.txt` | local + s3 | if you also use the S3/Kubernetes scripts |

Related files that must stay in sync (a unit test checks this):

* `../service/requirements.txt` – the pinned set installed into the Docker image
  (Linux only, so it keeps the plain `+cpu` pin).
* `../requirements-dev.txt` – used by the CI unit-test job (no PyTorch needed).

Supported Python: **3.10 – 3.12** (PyTorch 2.4.1 has no wheels for 3.13+). 3.11 is recommended.

To change a version, edit the pin in every file that lists the package, then run
`python -m pytest tests/test_requirements.py`.
