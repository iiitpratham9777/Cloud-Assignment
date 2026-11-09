"""Check that this machine is ready: Python version, dependencies, datasets, model weights.

  python scripts/check_setup.py                 # general check (missing data/weights are warnings)
  python scripts/check_setup.py --for serve     # need weights to run the API   (missing => FAIL)
  python scripts/check_setup.py --for train     # need data/train.jsonl to train (missing => FAIL)
  python scripts/check_setup.py --for test      # need only test dependencies

Standard library only, so it runs before anything is installed. Exit status 1 if any check FAILs.
Nothing is downloaded or modified.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from service import checks, settings  # noqa: E402

API_MODULES = ["fastapi", "uvicorn", "pydantic"]
ML_MODULES = ["torch", "transformers", "tokenizers", "safetensors", "numpy", "yaml"]
TEST_MODULES = ["pytest", "httpx", "fastapi", "yaml"]

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"


def run_checks(purpose="general", root=None, environ=None):
    """Return a list of (level, message). ``purpose``: general | serve | train | test."""
    env_backup = None
    if environ is not None:
        env_backup = dict(os.environ)
        os.environ.clear()
        os.environ.update(environ)
    try:
        return _run(purpose, root or settings.REPO_ROOT)
    finally:
        if env_backup is not None:
            os.environ.clear()
            os.environ.update(env_backup)


def _run(purpose, root):
    out = []
    problems = checks.check_python()
    out.append((FAIL, problems[0]) if problems else (OK, f"Python {sys.version.split()[0]}"))
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    out.append((OK, "running inside a virtual environment") if in_venv else
               (WARN, "not inside a virtual environment (run `python scripts/dev.py setup`, "
                      "then use `python scripts/dev.py ...`)"))

    wanted = {"serve": API_MODULES + ["torch", "transformers", "tokenizers", "safetensors"],
              "train": ML_MODULES,
              "test": TEST_MODULES,
              "general": sorted(set(API_MODULES + ML_MODULES + ["pytest"]))}[purpose]
    missing = checks.missing_modules(wanted)
    if missing:
        level = FAIL if purpose != "general" else WARN
        out.append((level, f"missing Python packages: {', '.join(missing)}  ->  "
                           "python scripts/dev.py setup"))
    else:
        out.append((OK, f"dependencies importable: {', '.join(wanted)}"))

    data_problems = checks.check_repo_data(root)
    out += [(FAIL, p) for p in data_problems] or [(OK, "repository data files present "
                                                       "(commands_pool, validation, schema)")]

    env_file = settings.resolve_path(os.environ.get("ENV_FILE") or ".env", root)
    out.append((OK, f"{env_file.name} found") if env_file.is_file() else
               (WARN, ".env not found (optional): copy .env.example to .env to change settings"))

    if purpose in ("train", "general"):
        tp = checks.check_train_file(settings.train_file() if root == settings.REPO_ROOT
                                     else settings.resolve_path("data/train.jsonl", root))
        level = FAIL if purpose == "train" else WARN
        out += [(level, t) for t in tp] or [(OK, "training dataset present")]
    if purpose in ("serve", "general"):
        if os.environ.get("MODEL_URI"):
            out.append((OK, f"MODEL_URI is set ({os.environ['MODEL_URI']}): the API will download the "
                            "model from object storage"))
        else:
            mp = checks.check_model_dir(settings.model_dir() if root == settings.REPO_ROOT
                                        else settings.resolve_path("model_out/v1", root))
            level = FAIL if purpose == "serve" else WARN
            out += [(level, m) for m in mp] or [(OK, f"model weights present at {settings.model_dir()}")]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--for", dest="purpose", choices=["general", "serve", "train", "test"],
                    default="general")
    a = ap.parse_args(argv)
    settings.load_dotenv()
    results = run_checks(a.purpose)
    for level, msg in results:
        first, *rest = msg.splitlines()
        print(f"[{level}] {first}")
        for line in rest:
            print(f"       {line}")
    failed = sum(1 for lvl, _ in results if lvl == FAIL)
    warned = sum(1 for lvl, _ in results if lvl == WARN)
    print(f"\n{failed} failure(s), {warned} warning(s).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
