"""Download pretrained model weights (a model-<version>.tar.gz archive) and unpack them.

Use this to run the API WITHOUT retraining, when someone has published weights, for example as a
GitHub Release asset (the archive format produced by `make train` / training/train.py).

  # from a GitHub Release of a repository you trust:
  python scripts/fetch_model.py --repo OWNER/REPO --tag model-release --version v1
  # from any URL (https:// or file://):
  python scripts/fetch_model.py --url https://example.org/model-v1.tar.gz --sha256 <64 hex chars> --version v1

The archive is verified against a SHA-256 checksum before it is unpacked:
  --sha256 HEX      explicit checksum, or
  models/<version>/model.sha256   (tracked in this repo, written by training/train.py).
Without a checksum the download is refused unless you pass --no-verify (not recommended).
Weights are unpacked into model_out/<version>/ (git-ignored). Existing folders are kept unless --force.
Only download weights from sources you trust: model files are loaded by PyTorch/Transformers.
"""
import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from service import checks  # noqa: E402
from service.settings import REPO_ROOT, resolve_path  # noqa: E402


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_expected_sha256(version, root=REPO_ROOT):
    """Checksum from models/<version>/model.sha256 ('<hex>  <filename>'), or None if absent."""
    path = os.path.join(root, "models", version, "model.sha256")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        parts = f.read().split()
    return parts[0].lower() if parts else None


def safe_extract(archive, dest):
    """Unpack a .tar.gz into ``dest``, refusing links, devices and paths that escape ``dest``."""
    dest = os.path.realpath(dest)
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            target = os.path.realpath(os.path.join(dest, m.name))
            if os.path.commonpath([dest, target]) != dest:
                raise ValueError(f"unsafe path in archive: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                raise ValueError(f"unsupported entry type in archive: {m.name!r}")
        for m in members:
            m.mode = 0o644 if m.isfile() else 0o755  # normalise permissions
            if hasattr(tarfile, "data_filter"):  # Python >= 3.10.12 / 3.11.4 / 3.12: extra safety net
                tar.extract(m, dest, filter="data")
            else:
                tar.extract(m, dest)
    return [m.name for m in members if m.isfile()]


def download(url, dest_file):
    req = urllib.request.Request(url, headers={"User-Agent": "cloud-command-parser-fetch-model"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_file, "wb") as out:
        shutil.copyfileobj(resp, out, 1 << 20)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help="GitHub repository as OWNER/REPO")
    ap.add_argument("--tag", default="model-release", help="GitHub Release tag (default: model-release)")
    ap.add_argument("--url", help="full URL of the .tar.gz archive (alternative to --repo)")
    ap.add_argument("--version", default="v1", help="model version name, e.g. v1 (default: v1)")
    ap.add_argument("--dest", help="target folder (default: model_out/<version>)")
    ap.add_argument("--sha256", help="expected SHA-256 of the archive")
    ap.add_argument("--no-verify", action="store_true", help="skip checksum verification (not recommended)")
    ap.add_argument("--force", action="store_true", help="replace an existing, non-empty target folder")
    a = ap.parse_args(argv)

    if bool(a.repo) == bool(a.url):
        sys.exit("ERROR: give exactly one of --repo OWNER/REPO or --url URL.")
    url = a.url or f"https://github.com/{a.repo}/releases/download/{a.tag}/model-{a.version}.tar.gz"
    dest = resolve_path(a.dest or os.path.join("model_out", a.version))
    if dest.is_dir() and any(dest.iterdir()) and not a.force:
        sys.exit(f"ERROR: {dest} already exists and is not empty. Use --force to replace it.")
    expected = (a.sha256 or read_expected_sha256(a.version) or "").lower() or None
    if not expected and not a.no_verify:
        sys.exit(f"ERROR: no checksum known for version {a.version!r}. Pass --sha256 HEX "
                 f"(published next to the weights) or, at your own risk, --no-verify.")

    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, f"model-{a.version}.tar.gz")
        print(f"downloading {url}")
        try:
            download(url, archive)
        except Exception as exc:  # network, 404, permission...
            sys.exit(f"ERROR: download failed: {exc}")
        digest = sha256_file(archive)
        if expected and digest != expected:
            sys.exit(f"ERROR: checksum mismatch.\n  expected {expected}\n  got      {digest}\n"
                     "Nothing was unpacked.")
        print(f"sha256 {digest}" + ("  (verified)" if expected else "  (NOT verified)"))
        if dest.is_dir() and a.force:
            shutil.rmtree(dest)
        try:
            files = safe_extract(archive, dest)
        except (ValueError, tarfile.TarError) as exc:
            shutil.rmtree(dest, ignore_errors=True)
            sys.exit(f"ERROR: could not unpack archive: {exc}")
    print(f"unpacked {len(files)} files to {dest}")
    problems = checks.check_model_dir(dest)
    if problems:
        sys.exit("WARNING: the archive unpacked but the folder does not look like a model:\n"
                 + "\n".join(problems))
    print(f"OK. Start the API with:  python scripts/dev.py serve   (MODEL_DIR={dest})")


if __name__ == "__main__":
    main()
