"""Sequence-level distillation: fine-tune Pythia-70M on the teacher's JSON replies (CPU).

  python -m training.train [--version v1] [--subset N]
Reads training/config.yaml (override with --config). Paths in the config are relative to the
repository root, so the command works from any folder on Windows, Linux and macOS.

Optional overrides for experiments and smoke tests (defaults come from the config):
  --train-file FILE   --base-model HF_ID_OR_DIR   --output-dir DIR   --loss-csv FILE
  --threads N         --save   (with --subset: also write the model; used by the smoke test)
"""
import argparse
import csv
import hashlib
import io
import json
import os
import random
import sys
import tarfile
import time

import yaml

from service import checks
from service.parsing import build_prompt, build_training_text, canonical_target
from service.settings import REPO_ROOT, resolve_path


def load_config(path):
    with open(resolve_path(path), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_examples(path, limit):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def write_loss(path, version, rows):
    """Keep one row per (version, epoch); rewrite the rows of this version only."""
    existing = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r["version"] != version]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "epoch", "train_loss"])
        for r in existing:
            w.writerow([r["version"], r["epoch"], r["train_loss"]])
        for epoch, loss in rows:
            w.writerow([version, epoch, f"{loss:.6f}"])


def package_dir(model_dir, version):
    """Deterministic tar.gz of the model dir; returns (archive path, sha256 of the archive)."""
    archive = os.path.join(os.path.dirname(model_dir), f"model-{version}.tar.gz")
    names = sorted(n for n in os.listdir(model_dir) if not n.endswith(".sha256"))
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for n in names:
            info = tar.gettarinfo(os.path.join(model_dir, n), arcname=n)
            info.mtime, info.uid, info.gid, info.uname, info.gname = 0, 0, 0, "", ""
            with open(os.path.join(model_dir, n), "rb") as fh:
                tar.addfile(info, fh)
    import gzip
    with open(archive, "wb") as out, gzip.GzipFile(fileobj=out, mode="wb", mtime=0) as gz:
        gz.write(raw.getvalue())
    with open(archive, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return archive, digest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--config", default="training/config.yaml")
    ap.add_argument("--train-file", help="override train_file from the config")
    ap.add_argument("--base-model", help="override base_model (Hugging Face id or local directory)")
    ap.add_argument("--output-dir", help="override output_dir from the config")
    ap.add_argument("--loss-csv", help="override loss_csv (or the subset loss file with --subset)")
    ap.add_argument("--threads", type=int, help="override threads from the config")
    ap.add_argument("--save", action="store_true",
                    help="with --subset, also save the model to <output_dir>/<version> (smoke tests)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.version not in cfg["versions"]:
        sys.exit(f"unknown version {a.version!r}; configured versions: {', '.join(cfg['versions'])}")
    vcfg = cfg["versions"][a.version]
    if a.train_file:
        cfg["train_file"] = a.train_file
    if a.base_model:
        cfg["base_model"] = a.base_model
    if a.output_dir:
        cfg["output_dir"] = a.output_dir
    if a.threads:
        cfg["threads"] = a.threads
    train_file = resolve_path(cfg["train_file"])
    try:
        checks.require_train_file(train_file)  # friendly message instead of a stack trace
    except checks.MissingArtifactError as exc:
        sys.exit(f"ERROR: {exc}")

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    want = cfg["library_versions"]
    for name, mod in (("torch", torch), ("transformers", transformers), ("numpy", np)):
        if not mod.__version__.startswith(str(want[name])):
            print(f"WARNING: {name} {mod.__version__} != configured {want[name]}", file=sys.stderr)

    seed = vcfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg["threads"])

    limit = a.subset or cfg["max_records"]
    limit = min(limit, cfg["max_records"])
    rows = load_examples(train_file, limit)
    print(f"training on {len(rows)} records, version {a.version}, seed {seed}", flush=True)

    base_model = str(resolve_path(cfg["base_model"])) if os.path.isdir(
        resolve_path(cfg["base_model"])) else cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model)

    examples = []
    for r in rows:
        target = canonical_target(r["teacher_raw"])
        prompt_ids = tok(build_prompt(r["text"]))["input_ids"]
        full_ids = tok(build_training_text(r["text"], target))["input_ids"] + [tok.eos_token_id]
        full_ids = full_ids[: cfg["max_seq_length"]]
        labels = list(full_ids)
        if cfg["mask_prompt_in_loss"]:
            labels[: len(prompt_ids)] = [-100] * min(len(prompt_ids), len(labels))
        examples.append((full_ids, labels))

    def batches(order):
        for i in range(0, len(order), cfg["batch_size"]):
            chunk = [examples[j] for j in order[i:i + cfg["batch_size"]]]
            width = max(len(ids) for ids, _ in chunk)
            ids = torch.full((len(chunk), width), tok.pad_token_id)
            lab = torch.full((len(chunk), width), -100)
            att = torch.zeros((len(chunk), width), dtype=torch.long)
            for k, (x, y) in enumerate(chunk):
                ids[k, :len(x)], lab[k, :len(y)], att[k, :len(x)] = (
                    torch.tensor(x), torch.tensor(y), 1)
            yield ids, att, lab

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    gen = random.Random(seed)
    losses = []
    model.train()
    for epoch in range(1, vcfg["epochs"] + 1):
        order = list(range(len(examples)))
        gen.shuffle(order)
        total, n, t0 = 0.0, 0, time.time()
        for ids, att, lab in batches(order):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            total += loss.item()
            n += 1
        losses.append((epoch, total / n))
        print(f"epoch {epoch}/{vcfg['epochs']} loss {total / n:.4f} ({time.time() - t0:.0f}s)",
              flush=True)

    if a.subset:
        subset_csv = resolve_path(a.loss_csv or "training/loss_subset.csv")
        os.makedirs(subset_csv.parent, exist_ok=True)
        write_loss(str(subset_csv), a.version, losses)
        if not a.save:
            print("SUBSET run finished (model not saved for release).")
            return
    else:
        loss_csv = resolve_path(a.loss_csv or cfg["loss_csv"])
        os.makedirs(loss_csv.parent, exist_ok=True)
        write_loss(str(loss_csv), a.version, losses)

    out = str(resolve_path(os.path.join(cfg["output_dir"], a.version)))
    os.makedirs(out, exist_ok=True)
    model.eval()
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    with open(os.path.join(out, "version.json"), "w", encoding="utf-8") as f:
        json.dump({"version": a.version, "base_model": cfg["base_model"], "seed": seed,
                   "epochs": vcfg["epochs"], "learning_rate": cfg["learning_rate"],
                   "batch_size": cfg["batch_size"], "records": len(rows),
                   "final_loss": losses[-1][1], "torch": torch.__version__,
                   "transformers": transformers.__version__, "smoke_subset": bool(a.subset)},
                  f, indent=2)
    archive, digest = package_dir(out, a.version)
    with open(os.path.join(out, "model.sha256"), "w") as f:
        f.write(f"{digest}  {os.path.basename(archive)}\n")
    if a.subset:  # smoke model: never touch the tracked checksums under models/
        print(f"SMOKE model saved to {out} (not a release candidate); sha256 {digest}")
        return
    checksum_dir = REPO_ROOT / "models" / a.version
    os.makedirs(checksum_dir, exist_ok=True)
    with open(checksum_dir / "model.sha256", "w") as f:
        f.write(f"{digest}  {os.path.basename(archive)}\n")
    print(f"saved {out}; release asset {archive}; sha256 {digest}")


if __name__ == "__main__":
    main()
