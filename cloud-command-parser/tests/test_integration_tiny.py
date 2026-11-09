"""End-to-end pipeline test with real PyTorch/Transformers on a TINY random model.

Verifies that the code runs: train -> save -> checksum/archive -> fetch -> load -> generate -> API.
It does NOT measure model quality and does not use Pythia-70M (that needs a download).
Skipped automatically when torch/transformers are not installed.
"""
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("SKIP_MODEL_LOAD", "1")

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_ML = True
except ImportError:
    HAVE_ML = False

if HAVE_ML:
    from scripts import fetch_model, make_smoke_dataset
    from service import checks
    from service.model_loader import Generator
    from tests.tiny_model import build_tiny_base_model, write_config
    from training import train


@unittest.skipUnless(HAVE_ML, "torch/transformers not installed")
class TinyPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        records = make_smoke_dataset.build(64)
        cls.train_file = cls.dir / "train.jsonl"
        cls.train_file.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        corpus = [r["text"] + " " + r["teacher_raw"] for r in records] + ["Command: JSON:"]
        cls.base = build_tiny_base_model(cls.dir / "base", corpus)
        cls.cfg = write_config(cls.dir / "config.json", cls.base, cls.train_file, cls.dir / "out",
                               cls.dir / "loss.csv")
        cls.repo = cls.dir / "fake_repo_root"  # so the full run never touches the real models/ folder

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_train(self, *extra):
        argv = ["train", "--config", str(self.cfg), "--version", "v1", *extra]
        out = io.StringIO()
        with mock.patch("sys.argv", argv), mock.patch.object(train, "REPO_ROOT", self.repo), \
                contextlib.redirect_stdout(out):
            train.main()
        return out.getvalue()

    def test_01_full_training_run_writes_model_checksum_and_loss(self):
        log = self.run_train()
        print("\n[tiny-pipeline] training log:\n" + log)
        model_dir = self.dir / "out" / "v1"
        self.assertEqual(checks.check_model_dir(model_dir), [])
        rows = (self.dir / "loss.csv").read_text(encoding="utf-8").splitlines()
        losses = [float(r.split(",")[2]) for r in rows[1:]]
        self.assertEqual(rows[0], "version,epoch,train_loss")
        self.assertEqual(len(losses), 30)
        self.assertLess(losses[-1], losses[0], "training loss should go down on the tiny dataset")
        archive = self.dir / "out" / "model-v1.tar.gz"
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        recorded = (self.repo / "models" / "v1" / "model.sha256").read_text(encoding="utf-8").split()
        self.assertEqual(recorded, [digest, "model-v1.tar.gz"])

    def test_02_archive_roundtrip_through_fetch_model_then_inference(self):
        if not (self.dir / "out" / "model-v1.tar.gz").exists():
            self.run_train()
        archive = self.dir / "out" / "model-v1.tar.gz"
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        dest = self.dir / "fetched"
        with contextlib.redirect_stdout(io.StringIO()):
            fetch_model.main(["--url", archive.as_uri(), "--sha256", digest, "--dest", str(dest)])
        self.assertEqual(checks.check_model_dir(dest), [])
        gen = Generator(dest, max_new_tokens=64)
        text = gen("what is the weather in london")
        self.assertIsInstance(text, str)
        print(f"\n[tiny-pipeline] generated: {text!r}")
        reference = Generator(self.dir / "out" / "v1", max_new_tokens=64)("what is the weather in london")
        self.assertEqual(text, reference, "fetched copy must behave identically (greedy decoding)")

    def test_03_api_serves_the_trained_tiny_model_from_local_dir(self):
        from fastapi.testclient import TestClient

        from service import app as app_module
        if not (self.dir / "out" / "v1").exists():
            self.run_train()
        app_module.state.update(generator=None, ready=False, error=None, detail=None)
        with mock.patch.dict(os.environ, {"MODEL_DIR": str(self.dir / "out" / "v1")}):
            os.environ.pop("MODEL_URI", None)
            app_module._load()
        try:
            client = TestClient(app_module.app)
            self.assertEqual(client.get("/readyz").status_code, 200)
            r = client.post("/parse", json={"text": "what is the weather in london"})
            print(f"\n[tiny-pipeline] API answered HTTP {r.status_code}: {r.text}")
            self.assertIn(r.status_code, (200, 500))
            if r.status_code == 500:
                self.assertEqual(r.json(), {"error": "invalid_model_output"})
            else:
                self.assertEqual(list(r.json()), ["intent", "date", "time", "place"])
        finally:
            app_module.state.update(generator=None, ready=False, error=None, detail=None)

    def test_04_subset_save_never_touches_tracked_checksums(self):
        out_before = set((self.dir / "out").glob("*"))
        self.run_train("--subset", "16", "--save", "--output-dir", str(self.dir / "smoke_out"),
                       "--loss-csv", str(self.dir / "smoke_loss.csv"))
        self.assertTrue((self.dir / "smoke_out" / "v1" / "model.safetensors").exists())
        version_json = self.dir / "smoke_out" / "v1" / "version.json"
        self.assertTrue(json.loads(version_json.read_text(encoding="utf-8"))["smoke_subset"])
        self.assertEqual(set((self.dir / "out").glob("*")), out_before)

    def test_05_missing_dataset_stops_with_clear_message(self):
        argv = ["train", "--config", str(self.cfg), "--train-file", str(self.dir / "nope.jsonl")]
        with mock.patch("sys.argv", argv), self.assertRaises(SystemExit) as cm:
            train.main()
        self.assertIn("Training dataset not found", str(cm.exception))
        self.assertIn("label_local.py", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
