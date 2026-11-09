"""Scan the files that would be committed for secrets and personal/local paths.

  python scripts/audit_repo.py            # scan; exit status 1 if anything suspicious is found
  python scripts/audit_repo.py --list     # also list every file that was scanned

Scanned: files tracked by Git plus untracked files that are NOT ignored (what `git add -A` would
add). Without Git, the whole tree is walked, skipping common cache/venv folders. It looks for
private keys, cloud/GitHub/Hugging Face/Anthropic/OpenAI-style tokens, `.env` files that are not
examples, absolute home-directory paths, and model weight files. It is a safety net, not a
guarantee: it cannot find every possible secret. CI additionally runs gitleaks.

To silence a deliberate example, put `audit:allow` on the same line.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Patterns are assembled from pieces so that this file does not flag itself.
_P = "".join
PATTERNS = [
    ("private key", re.compile(_P(["-----BEGIN ", r"[A-Z ]*", "PRIVATE KEY-----"]))),
    ("AWS access key id", re.compile(_P([r"\b(?:AK", "IA|AS", r"IA)[0-9A-Z]{16}\b"]))),
    ("GitHub token", re.compile(
        _P([r"\b(?:gh[pousr]", r"_[A-Za-z0-9]{30,}|github", r"_pat_[A-Za-z0-9_]{30,})"]))),
    ("Hugging Face token", re.compile(_P([r"\bhf", r"_[A-Za-z0-9]{30,}\b"]))),
    ("Anthropic/OpenAI-style key", re.compile(_P([r"\bsk-(?:ant-)?", r"[A-Za-z0-9_-]{30,}"]))),
    ("Slack token", re.compile(_P([r"\bxox[baprs]", r"-[A-Za-z0-9-]{10,}"]))),
    ("personal path (Windows)",
     re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+(?!Public\b|<|YOU|you|name|USERNAME)[^\\/\s\"'<>]+",
                re.IGNORECASE)),
    ("personal path (macOS)", re.compile(r"/Users/(?!Shared\b|<|YOU|you|name|USERNAME)[A-Za-z0-9._-]+/")),
    ("personal path (Linux)",
     re.compile(r"/home/(?!runner\b|<|YOU|you|name|USERNAME|user\b)[a-z][a-z0-9._-]*/")),
    ("secret assigned a literal value",
     re.compile(r"(?i)\b(?:password|passwd|secret_?key|api_?key|access_?key|auth_?token)\b\s*[:=]\s*"
                r"[\"']?(?!<|\$|\{|your|changeme|example|xxx|none|null|true|false|\"|')[A-Za-z0-9/+_\-]{12,}")),
]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules",
             "model_out", ".secrets"}
BINARY_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".gz", ".pyc", ".ico"}
WEIGHT_EXT = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".gguf"}
ENV_EXAMPLE = {".env.example", ".env.sample", ".env.template"}


def candidate_files(root):
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                             cwd=root, capture_output=True, check=True).stdout
        names = [n for n in out.decode("utf-8", "replace").split("\0") if n]
        return [root / n for n in names if (root / n).is_file()], "git"
    except (OSError, subprocess.CalledProcessError):
        pass
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        # Without Git we cannot know .gitignore, so local config that is normally ignored is skipped.
        files += [Path(dirpath) / f for f in filenames if f != ".env"]
    return files, "walk"


def scan_text(text):
    """Yield (line_number, label, snippet) for suspicious lines in ``text``."""
    for n, line in enumerate(text.splitlines(), 1):
        if "audit:allow" in line:
            continue
        for label, rx in PATTERNS:
            m = rx.search(line)
            if m:
                snippet = m.group(0)
                yield n, label, snippet[:6] + "..." if len(snippet) > 12 else snippet


def scan(root=ROOT):
    findings = []
    files, mode = candidate_files(Path(root))
    for path in files:
        rel = path.relative_to(root).as_posix()
        name = path.name
        if (name == ".env" or (name.startswith(".env.") and name not in ENV_EXAMPLE)
                or name.endswith(".env")):
            findings.append((rel, 0, "env file would be committed", name))
            continue
        if path.suffix.lower() in WEIGHT_EXT or name.endswith(".tar.gz"):
            findings.append((rel, 0, "model weights / large binary would be committed", name))
            continue
        if path.suffix.lower() in BINARY_EXT:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        findings += [(rel, n, label, snip) for n, label, snip in scan_text(text)]
    return findings, files, mode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list scanned files")
    a = ap.parse_args(argv)
    findings, files, mode = scan()
    print(f"scanned {len(files)} files ({'git-aware' if mode == 'git' else 'plain walk: no Git info'})")
    if a.list:
        for f in sorted(files):
            print("  ", f.relative_to(ROOT).as_posix())
    for rel, n, label, snip in findings:
        where = f"{rel}:{n}" if n else rel
        print(f"SUSPICIOUS  {where}  [{label}]  {snip}")
    if findings:
        print(f"\n{len(findings)} suspicious item(s). Remove them, add the file to .gitignore, or mark a "
              "deliberate example with `audit:allow`.")
        return 1
    print("nothing suspicious found (this check cannot prove the absence of secrets).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
