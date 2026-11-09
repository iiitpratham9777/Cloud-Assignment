"""One cross-platform entry point for everyday tasks (Windows, Linux, macOS). Standard library only.

  python scripts/dev.py setup [--no-ml] [--with-s3] [--recreate]   create .venv, install, copy .env
  python scripts/dev.py check [serve|train|test]                    verify deps, data and weights
  python scripts/dev.py serve [--host H] [--port P] [--reload]      run the FastAPI API locally
  python scripts/dev.py test  [pytest args...]                      run the unit tests
  python scripts/dev.py lint                                        ruff check
  python scripts/dev.py train [--version v1] [--subset N] ...       fine-tune (args go to training.train)
  python scripts/dev.py smoke [--base-model ID_OR_DIR]              tiny end-to-end training check
  python scripts/dev.py audit                                       look for secrets / personal paths

Run it with any Python 3.10-3.12 (Windows: `py -3.11 scripts\\dev.py setup`). Every command except
`setup` uses the project's .venv automatically, so you never have to "activate" it.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from service import checks, settings  # noqa: E402

VENV = ROOT / ".venv"


def venv_python():
    """Interpreter of the project venv if it exists, else the interpreter running this script."""
    candidate = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(candidate) if candidate.exists() else sys.executable


def venv_missing(modules):
    """Modules that the interpreter we will actually use (the venv's) cannot import."""
    code = ("import importlib.util, sys; "
            "print(' '.join(m for m in sys.argv[1:] if importlib.util.find_spec(m) is None))")
    out = subprocess.run([venv_python(), "-c", code, *modules], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.split() if out.returncode == 0 else list(modules)


def run(cmd, check=True, **kw):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw)
    if check and proc.returncode != 0:
        sys.exit(proc.returncode)
    return proc.returncode


def cmd_setup(a):
    problems = checks.check_python()
    if problems:
        sys.exit(f"ERROR: {problems[0]}\nInstall a supported Python (https://www.python.org/downloads/) "
                 "and re-run with it, e.g. `py -3.11 scripts\\dev.py setup` on Windows or "
                 "`python3.11 scripts/dev.py setup` on Linux/macOS.")
    if a.recreate and VENV.exists():
        shutil.rmtree(VENV)
    if not VENV.exists():
        try:
            run([sys.executable, "-m", "venv", VENV])
        except SystemExit:
            sys.exit("ERROR: could not create the virtual environment. On Debian/Ubuntu install the "
                     "venv module first: `sudo apt install python3-venv`.")
    py = venv_python()
    run([py, "-m", "pip", "install", "--upgrade", "pip"])
    if a.no_ml:
        req = ROOT / "requirements-dev.txt"  # API + tests only, no PyTorch (small and fast)
    else:
        req = ROOT / "requirements" / ("all.txt" if a.with_s3 else "local.txt")
    run([py, "-m", "pip", "install", "-r", req])
    env_example, env_file = ROOT / ".env.example", ROOT / ".env"
    if env_example.exists() and not env_file.exists():
        shutil.copyfile(env_example, env_file)
        print("created .env from .env.example (edit it to change settings; it is git-ignored)")
    print("\nSetup finished. Next steps:")
    run([py, "scripts/check_setup.py"], check=False)
    print("\nThen:  python scripts/dev.py test   |   python scripts/dev.py serve")


def cmd_check(a):
    sys.exit(run([venv_python(), "scripts/check_setup.py", "--for", a.purpose], check=False))


def cmd_serve(a):
    settings.load_dotenv()
    host = a.host or os.environ.get("HOST") or "127.0.0.1"
    port = str(a.port or settings.env_int("PORT", 8000))
    if not os.environ.get("MODEL_URI"):
        problems = checks.check_model_dir(settings.model_dir())
        if problems and not a.allow_no_model:
            sys.exit("ERROR: cannot start with a model yet.\n" + "\n".join(problems) +
                     "\n(Use --allow-no-model to start the API anyway; /parse will answer 503.)")
    missing = venv_missing(["uvicorn", "fastapi"])
    if missing:
        sys.exit(f"ERROR: missing packages ({', '.join(missing)}). Run: python scripts/dev.py setup")
    print(f"API docs will be at http://{host}:{port}/docs  (Ctrl+C to stop)")
    cmd = [venv_python(), "-m", "uvicorn", "service.app:app", "--host", host, "--port", port]
    if a.reload:
        cmd.append("--reload")
    sys.exit(run(cmd, check=False))


def cmd_test(a):
    sys.exit(run([venv_python(), "-m", "pytest", "-q", *a.args], check=False))


def cmd_lint(a):
    sys.exit(run([venv_python(), "-m", "ruff", "check", "."], check=False))


def cmd_train(a):
    sys.exit(run([venv_python(), "-m", "training.train", *a.args], check=False))


def cmd_smoke(a):
    """Tiny end-to-end training run on SYNTHETIC data. Proves the code runs; proves nothing about quality."""
    py = venv_python()
    run([py, "scripts/make_smoke_dataset.py"])
    cmd = [py, "-m", "training.train", "--version", "v1", "--subset", "32", "--save",
           "--train-file", "data/smoke_train.jsonl", "--output-dir", "model_out/smoke",
           "--loss-csv", "model_out/smoke/loss_smoke.csv"]
    if a.base_model:
        cmd += ["--base-model", a.base_model]
    run(cmd)
    print("\nSmoke model written to model_out/smoke/v1 (synthetic data; not a real model).")
    print("Try it:  MODEL_DIR=model_out/smoke/v1 python scripts/dev.py serve")


def cmd_audit(a):
    sys.exit(run([sys.executable, "scripts/audit_repo.py", *a.args], check=False))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    # These commands forward every remaining argument untouched (flags included) to another tool.
    passthrough = {"test": cmd_test, "train": cmd_train, "audit": cmd_audit}
    if argv and argv[0] in passthrough:
        return passthrough[argv[0]](argparse.Namespace(args=argv[1:]))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup", help="create .venv and install dependencies")
    p.add_argument("--no-ml", action="store_true", help="skip PyTorch/transformers (API + tests only)")
    p.add_argument("--with-s3", action="store_true", help="also install boto3 (Kubernetes/S3 workflow)")
    p.add_argument("--recreate", action="store_true", help="delete and recreate .venv first")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("check", help="verify dependencies, datasets and model weights")
    p.add_argument("purpose", nargs="?", default="general", choices=["general", "serve", "train", "test"])
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("serve", help="run the FastAPI service locally (no Docker/Kubernetes)")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--reload", action="store_true", help="auto-reload on code changes (development)")
    p.add_argument("--allow-no-model", action="store_true", help="start even if weights are missing")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("test", help="run unit tests (extra arguments go to pytest)")
    p.set_defaults(fn=cmd_test)

    p = sub.add_parser("lint", help="run ruff")
    p.set_defaults(fn=cmd_lint)

    p = sub.add_parser("train", help="fine-tune the model (extra arguments go to training.train)")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("smoke", help="tiny end-to-end training check on synthetic data")
    p.add_argument("--base-model", help="Hugging Face id or local folder (default: base_model in config)")
    p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("audit", help="scan for secrets and personal paths before publishing")
    p.set_defaults(fn=cmd_audit)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
