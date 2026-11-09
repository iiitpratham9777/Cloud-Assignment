# Attribution

 
| Role | Name |
|---|---|
| Student working  | **Pratham Gupta** (roll number 230103034) |

 
1. keep this file and the credit lines in `README.md`, `STATUS.md`, `AI_USE.md`, `report/report.md` and
   `reflection/reflection_draft.md`;
2. add yourself in the table above (for example "Customised by <your name>, <year>") - do not replace existing rows;
3. describe your changes in your own README or changelog;
4. do not present measurements, reports or reflections produced by someone else as your own.

The student roll number is an academic identifier from the original working copy and is kept as a credit. If you are publishing
a copy where that should not be public, ask the person it belongs to before deciding.

## AI assistance

`AI_USE.md` records which AI tools were used on this project. The later portability work (settings, checks, local mode, tooling,
tests, documentation - see `docs/CHANGES.md`) was also done with AI assistance (Claude, Anthropic); the person who publishes the
result should review it and keep `AI_USE.md` truthful.

## Provenance of included data and tools (needs confirmation)

| Item | What the repository says | Status |
|---|---|---|
| `data/commands_pool.jsonl` | README: "3,000 SLURP commands" | Plausibly derived from SLURP text (see below); exact derivation **not verified** |
| `data/validation.jsonl`, `data/schema.json`, `data/rate_card.json`, `tools/score.py`, `tools/locustfile.py`, `tools/nacl_sim.py`, `network/packets_sample.jsonl` | README: "provided kit, unchanged" | Author/licence of this kit **unknown**; confirm with whoever supplied it before redistributing |
| Everything else (service, worker, training, scripts, k8s, tests, docs) | written for this project | covered by the project licence once one is granted - see `LICENSING.md` |

## Third-party material (not bundled; downloaded or installed separately)

Licences below were checked on 9 November 2026 against the sources named. Always re-check the source before redistributing.

| Component | Used for | Licence | Source checked |
|---|---|---|---|
| SLURP dataset (Bastianelli et al., EMNLP 2020) | origin of the command texts | text annotations **CC BY 4.0** (attribution required); audio **CC BY-NC 4.0** (not used here) | SLURP repository README (github.com/pswietojanski/slurp) |
| EleutherAI Pythia-70M | base model that is fine-tuned | Apache 2.0 | Pythia model card text (as mirrored on Hugging Face) |
| Microsoft Phi-3.5-mini-instruct (via Ollama `phi3.5`) | teacher model for labelling | MIT | third-party listings quoting the licence; confirm on the official `microsoft/Phi-3.5-mini-instruct` page |
| PyTorch 2.4.1 | training/inference | BSD-style (metadata: BSD-3) | installed package metadata |
| Transformers 4.44.2, tokenizers 0.19.1, huggingface-hub 0.24.7, safetensors 0.4.5, boto3 1.35.20 | model stack, S3 | Apache 2.0 | installed package metadata |
| NumPy 1.26.4, Starlette 0.38.6, uvicorn 0.30.6, httpx 0.27.2 | numerics, web stack, tests | BSD | installed package metadata |
| FastAPI 0.115.0, pydantic 2.9.2, PyYAML 6.0.2, pytest 8.3.3, ruff 0.6.9 | API, validation, config, tests, lint | MIT | installed package metadata |
| SeaweedFS, k3d/k3s, Ollama, Locust, gitleaks | optional tooling | **not checked** - they are not bundled; read their licences if you redistribute images or binaries | - |

Because the weights you train are fine-tuned from Pythia-70M and the labels come from Phi-3.5-mini, keep the Apache 2.0 and MIT
notices if you redistribute trained weights, and credit SLURP (CC BY 4.0) when you redistribute data derived from it:

> Bastianelli, E., Vanzo, A., Swietojanski, P., Rieser, V. (2020). *SLURP: A Spoken Language Understanding Resource Package.* EMNLP 2020.

This file is documentation, not legal advice.
