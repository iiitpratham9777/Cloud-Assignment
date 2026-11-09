"""Label commands with the teacher model (Ollama) on this machine: no Kubernetes, no S3.

  python scripts/label_local.py                    # all 3,000 commands -> results/teacher/shard-0.jsonl
  python scripts/label_local.py --limit 50         # quick trial run
  python scripts/label_local.py --shard-index 1 --shard-count 2   # run half on this laptop, half on another

Then clean the raw replies into the training file:
  python -m worker.clean --local results/teacher --out data     # -> data/train.jsonl

Uses exactly the same prompt, request and sharding rule as the Kubernetes worker
(worker/worker.py), so the output is interchangeable. The run is resumable: re-run the same
command and finished ids are skipped. Labelling 3,000 commands on a CPU laptop can take hours.

Settings (flags override environment variables / .env):
  OLLAMA_URL (default http://localhost:11434)   TEACHER_MODEL (default phi3.5)
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from service import settings  # noqa: E402
from worker.worker import ask_teacher, select_shard  # noqa: E402


def ollama_models(url):
    """Names of the models the Ollama server has pulled; raises URLError if unreachable."""
    with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=10) as resp:
        return [m.get("name", "") for m in json.load(resp).get("models", [])]


def model_available(names, model):
    """True if ``model`` is among the Ollama model names ('phi3.5' matches 'phi3.5:latest')."""
    if ":" in model:
        return model in names
    return any(n == model or n.split(":")[0] == model for n in names)


def read_ids(path):
    ids = set()
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ids.add(json.loads(line)["id"])
    return ids


def main():
    settings.load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="data/commands_pool.jsonl")
    ap.add_argument("--out-dir", default="results/teacher")
    ap.add_argument("--prompt", default="prompts/teacher_prompt.txt")
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_URL", "http://localhost:11434"))
    ap.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "phi3.5"))
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="label at most N commands (0 = all in the shard)")
    a = ap.parse_args()

    try:
        names = ollama_models(a.ollama)
    except (urllib.error.URLError, OSError) as exc:
        sys.exit(f"ERROR: cannot reach Ollama at {a.ollama} ({exc}).\n"
                 "Install Ollama from https://ollama.com, start it (`ollama serve`), then run "
                 f"`ollama pull {a.model}`.")
    if not model_available(names, a.model):
        sys.exit(f"ERROR: Ollama has no model named {a.model!r} (found: {', '.join(names) or 'none'}).\n"
                 f"Run: ollama pull {a.model}")

    pool = settings.resolve_path(a.pool)
    prompt_path = settings.resolve_path(a.prompt)
    if not pool.is_file():
        sys.exit(f"ERROR: command pool not found: {pool}")
    with open(pool, encoding="utf-8") as f:
        lines = f.read().splitlines()
    try:
        mine = select_shard(lines, a.shard_index, a.shard_count)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()

    out_dir = settings.resolve_path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard-{a.shard_index}.jsonl"
    done = read_ids(out_path)
    todo = [json.loads(ln) for ln in mine if ln.strip() and json.loads(ln)["id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"shard {a.shard_index}/{a.shard_count}: {len(mine)} commands, {len(done)} already done, "
          f"{len(todo)} to label with {a.model} -> {out_path}", flush=True)
    with open(out_path, "a", encoding="utf-8", newline="\n") as out:
        for n, rec in enumerate(todo, 1):
            raw = ask_teacher(rec["text"], prompt, a.ollama, a.model)
            out.write(json.dumps({"id": rec["id"], "text": rec["text"], "teacher_raw": raw},
                                 ensure_ascii=False) + "\n")
            out.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} labelled", flush=True)
    print(f"finished. Next: python -m worker.clean --local {a.out_dir} --out data")


if __name__ == "__main__":
    main()
