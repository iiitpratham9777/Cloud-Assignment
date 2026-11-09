"""Pre-flight checks for datasets, model weights and Python dependencies.

Standard library only. Used by ``scripts/check_setup.py``, ``training/train.py`` and the API so
that a missing file produces one clear message that says what is missing and how to fix it,
instead of a stack trace from deep inside PyTorch or Transformers.
"""
import importlib.util
import json
import sys
from pathlib import Path

from .settings import REPO_ROOT

SUPPORTED_PYTHON = ((3, 10), (3, 12))  # inclusive; pinned torch 2.4.1 has no wheels for 3.13+

DATA_HELP = (
    "How to get a training dataset (see README, section 'Training data'):\n"
    "  1. Already have a cleaned data/train.jsonl from this project? Copy it to the path above.\n"
    "  2. Generate it with the teacher model: install Ollama, `ollama pull phi3.5`, then run\n"
    "       python scripts/label_local.py && python -m worker.clean --local results/teacher --out data\n"
    "  3. Only want to test that the training code runs? `python scripts/make_smoke_dataset.py`\n"
    "     (synthetic, tiny; NOT suitable for a real model)."
)

MODEL_HELP = (
    "How to get model weights (see README, section 'Model weights'):\n"
    "  1. Train one:            python scripts/dev.py train     (needs data/train.jsonl)\n"
    "  2. Download a release:   python scripts/fetch_model.py --repo OWNER/REPO --tag TAG --version v1\n"
    "  3. Already have a model folder? Set MODEL_DIR=/path/to/folder in .env"
)


class MissingArtifactError(RuntimeError):
    """A required dataset or model directory is missing or incomplete (message is user-facing)."""


def check_python(version=None):
    """Return a list of problems for the running Python version (empty = fine)."""
    v = tuple((version or sys.version_info)[:2])
    lo, hi = SUPPORTED_PYTHON
    if not lo <= v <= hi:
        return [f"Python {v[0]}.{v[1]} is not supported; use Python {lo[0]}.{lo[1]} to {hi[0]}.{hi[1]} "
                f"(3.11 recommended)."]
    return []


def missing_modules(names):
    """Names from ``names`` that cannot be imported (checked without importing them)."""
    out = []
    for n in names:
        try:
            found = importlib.util.find_spec(n) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(n)
    return out


def _count_jsonl(path, required_keys, sample=50):
    """Return (record_count, problem_or_None); validates the first ``sample`` records."""
    count = 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            count += 1
            if count <= sample:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return count, f"line {n} is not valid JSON"
                missing = [k for k in required_keys if not isinstance(rec, dict) or k not in rec]
                if missing:
                    return count, f"line {n} is missing key(s): {', '.join(missing)}"
    return count, None


def check_repo_data(root=REPO_ROOT):
    """Files that ship with the repository (inputs for labelling, evaluation and the API)."""
    problems = []
    root = Path(root)
    for rel, keys in (("data/commands_pool.jsonl", ("id", "text")),
                      ("data/validation.jsonl", ("id", "text", "intent", "date", "time", "place"))):
        p = root / rel
        if not p.is_file():
            problems.append(f"{rel} is missing (it ships with the repository; re-download the project "
                            f"or run `git checkout {rel}`).")
            continue
        n, bad = _count_jsonl(p, keys)
        if bad or n == 0:
            problems.append(f"{rel} is unreadable: {bad or 'file is empty'}.")
    if not (root / "data" / "schema.json").is_file():
        problems.append("data/schema.json is missing (it ships with the repository).")
    return problems


def check_train_file(path):
    """Problems with the cleaned teacher dataset used for training (empty list = usable)."""
    p = Path(path)
    if not p.is_file():
        return [f"Training dataset not found: {p}\n{DATA_HELP}"]
    n, bad = _count_jsonl(p, ("id", "text", "teacher_raw"))
    if n == 0:
        return [f"Training dataset is empty: {p}\n{DATA_HELP}"]
    if bad:
        return [f"Training dataset {p} is malformed: {bad}. Expected JSON Lines with keys "
                f"id, text, teacher_raw (the output of `python -m worker.clean`)."]
    return []


_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json",
                 "pytorch_model.bin.index.json")
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json")


def check_model_dir(path):
    """Problems with a local model directory (empty list = looks loadable)."""
    p = Path(path)
    if not p.is_dir():
        return [f"Model directory not found: {p}\n{MODEL_HELP}"]
    problems = []
    if not (p / "config.json").is_file():
        problems.append("config.json")
    if not any((p / w).is_file() for w in _WEIGHT_FILES):
        problems.append("weights (model.safetensors or pytorch_model.bin)")
    if not any((p / t).is_file() for t in _TOKENIZER_FILES):
        problems.append("tokenizer files (tokenizer.json)")
    if problems:
        return [f"Model directory {p} is incomplete; missing: {', '.join(problems)}.\n{MODEL_HELP}"]
    return []


def require_train_file(path):
    problems = check_train_file(path)
    if problems:
        raise MissingArtifactError("\n".join(problems))


def require_model_dir(path):
    problems = check_model_dir(path)
    if problems:
        raise MissingArtifactError("\n".join(problems))
