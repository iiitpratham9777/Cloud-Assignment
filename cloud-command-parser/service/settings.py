"""Environment-driven configuration shared by the API, training and helper scripts.

Standard library only, so it can be imported before any ML dependency is installed.

Rules
  * Real environment variables always win.
  * A ``.env`` file in the repository root (copy of ``.env.example``) is read if it exists;
    it never overrides variables that are already set. Point ``ENV_FILE`` at another file
    to use a different one.
  * Relative paths are resolved against the repository root, never against the current
    working directory, so commands work from any folder and on every operating system.
"""
import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_DIR = "model_out/v1"
DEFAULT_TRAIN_FILE = "data/train.jsonl"


def resolve_path(value, base=REPO_ROOT):
    """Expand ``~`` and make ``value`` absolute (relative values are relative to ``base``)."""
    p = Path(os.path.expanduser(str(value)))
    return p if p.is_absolute() else Path(base) / p


def parse_dotenv(text):
    """Parse the text of a .env file into a dict. Supports KEY=VALUE, quotes, comments, 'export '."""
    values = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not key:
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
            val = val[1:-1]
        elif " #" in val:  # inline comment on an unquoted value
            val = val.split(" #", 1)[0].rstrip()
        values[key] = val
    return values


def load_dotenv(path=None):
    """Load a .env file into os.environ without overriding existing variables.

    Blank values (``KEY=``) are skipped, so an empty placeholder never shadows a real setting such
    as credentials from ``~/.aws``. Returns the dict of variables actually set ({} if no file)."""
    env_path = resolve_path(path or os.environ.get("ENV_FILE") or ".env")
    if not env_path.is_file():
        return {}
    applied = {}
    for key, val in parse_dotenv(env_path.read_text(encoding="utf-8")).items():
        if val != "" and key not in os.environ:
            os.environ[key] = val
            applied[key] = val
    return applied


def model_dir():
    """Local directory holding a trained model (config.json, weights, tokenizer files)."""
    return resolve_path(os.environ.get("MODEL_DIR") or DEFAULT_MODEL_DIR)


def model_cache_dir():
    """Where a model downloaded from object storage is cached (portable temp dir by default)."""
    configured = os.environ.get("MODEL_CACHE_DIR")
    if configured:
        return resolve_path(configured)
    return Path(tempfile.gettempdir()) / "command-parser-model"


def train_file():
    return resolve_path(os.environ.get("TRAIN_FILE") or DEFAULT_TRAIN_FILE)


def env_int(name, default):
    """Integer environment variable with a clear error instead of a bare ValueError."""
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"Environment variable {name}={raw!r} must be an integer.") from None
