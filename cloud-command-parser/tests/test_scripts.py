import contextlib
import hashlib
import http.server
import io
import json
import os
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from scripts import audit_repo, check_setup, fetch_model, label_local, make_smoke_dataset
from worker.clean import clean


def make_archive(path, files, extra=None):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if extra:
            extra(tar)


MODEL_FILES = {"config.json": b"{}", "model.safetensors": b"w", "tokenizer.json": b"{}"}


class FetchModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def archive(self, **kw):
        a = self.dir / "model-v1.tar.gz"
        make_archive(a, MODEL_FILES, **kw)
        return a

    def test_safe_extract_good_archive(self):
        files = fetch_model.safe_extract(self.archive(), self.dir / "out")
        self.assertEqual(sorted(files), sorted(MODEL_FILES))
        self.assertTrue((self.dir / "out" / "config.json").is_file())

    def test_safe_extract_rejects_path_traversal(self):
        a = self.dir / "evil.tar.gz"
        make_archive(a, {"../escape.txt": b"x"})
        with self.assertRaises(ValueError):
            fetch_model.safe_extract(a, self.dir / "out")
        self.assertFalse((self.dir / "escape.txt").exists())

    def test_safe_extract_rejects_symlinks(self):
        def add_link(tar):
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
        a = self.dir / "link.tar.gz"
        make_archive(a, {}, extra=add_link)
        with self.assertRaises(ValueError):
            fetch_model.safe_extract(a, self.dir / "out")

    def run_main(self, args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                fetch_model.main(args)
                code = 0
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 1
                if isinstance(exc.code, str):
                    err.write(exc.code)
        return code, out.getvalue() + err.getvalue()

    def test_end_to_end_with_checksum(self):
        a = self.archive()
        digest = hashlib.sha256(a.read_bytes()).hexdigest()
        dest = self.dir / "dest"
        code, text = self.run_main(["--url", a.as_uri(), "--sha256", digest, "--dest", str(dest)])
        self.assertEqual(code, 0, text)
        self.assertIn("(verified)", text)
        self.assertTrue((dest / "model.safetensors").is_file())

    def test_checksum_mismatch_unpacks_nothing(self):
        a = self.archive()
        dest = self.dir / "dest"
        code, text = self.run_main(["--url", a.as_uri(), "--sha256", "0" * 64, "--dest", str(dest)])
        self.assertNotEqual(code, 0)
        self.assertIn("checksum mismatch", text)
        self.assertFalse(dest.exists())

    def test_refuses_without_checksum(self):
        a = self.archive()
        code, text = self.run_main(["--url", a.as_uri(), "--version", "v-no-such", "--dest",
                                    str(self.dir / "d")])
        self.assertNotEqual(code, 0)
        self.assertIn("no checksum", text)

    def test_refuses_to_overwrite_without_force(self):
        a = self.archive()
        dest = self.dir / "dest"
        dest.mkdir()
        (dest / "keep.txt").write_text("x", encoding="utf-8")
        code, text = self.run_main(["--url", a.as_uri(), "--no-verify", "--dest", str(dest)])
        self.assertNotEqual(code, 0)
        self.assertIn("--force", text)

    def test_needs_exactly_one_source(self):
        code, _ = self.run_main([])
        self.assertNotEqual(code, 0)

    def test_repo_checksum_file_is_read(self):
        root = self.dir / "root"
        (root / "models" / "v9").mkdir(parents=True)
        (root / "models" / "v9" / "model.sha256").write_text("ABC123  model-v9.tar.gz\n", encoding="utf-8")
        self.assertEqual(fetch_model.read_expected_sha256("v9", root), "abc123")
        self.assertIsNone(fetch_model.read_expected_sha256("v10", root))


class CheckSetupTests(unittest.TestCase):
    def test_serve_fails_without_weights_and_train_fails_without_data(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data").mkdir()
            env = {"PATH": os.environ.get("PATH", "")}
            serve = check_setup.run_checks("serve", root=root, environ=env)
            self.assertTrue(any(lvl == "FAIL" and "Model directory" in m for lvl, m in serve))
            train = check_setup.run_checks("train", root=root, environ=env)
            self.assertTrue(any(lvl == "FAIL" and "Training dataset" in m for lvl, m in train))

    def test_general_downgrades_missing_artifacts_to_warnings(self):
        with tempfile.TemporaryDirectory() as d:
            res = check_setup.run_checks("general", root=Path(d), environ={"PATH": ""})
            weights = [r for r in res if "Model directory" in r[1] or "Training dataset" in r[1]]
            self.assertEqual({lvl for lvl, _ in weights}, {"WARN"})

    def test_model_uri_counts_as_weights_source(self):
        with tempfile.TemporaryDirectory() as d:
            res = check_setup.run_checks("serve", root=Path(d),
                                         environ={"MODEL_URI": "s3://models/v1/"})
            self.assertFalse(any("Model directory" in m for _, m in res))


class AuditTests(unittest.TestCase):
    def test_detects_fabricated_secrets_and_personal_paths(self):
        # Fake values are assembled at run time so this file itself stays clean.
        fake = {
            "aws": "AK" + "IA" + "ABCDEFGHIJKLMNOP",
            "gh": "gh" + "p_" + "a" * 36,
            "hf": "hf" + "_" + "b" * 34,
            "key": "-----BEGIN " + "RSA PRIVATE KEY-----",
            "win": "C:" + "\\Users\\" + "somebody\\project",
            "mac": "/Users/" + "somebody/project",
            "linux": "/home/" + "somebody/project",
            "assign": "api_key = " + "Zx9" + "Qw8" + "Er7Ty6Ui5",
        }
        for name, value in fake.items():
            hits = list(audit_repo.scan_text(f"x = '{value}'"))
            self.assertTrue(hits, f"{name} was not detected")

    def test_placeholders_and_allow_marker_are_not_flagged(self):
        clean_lines = [
            "AWS_ACCESS_KEY_ID=",
            "AWS_SECRET_ACCESS_KEY: \"<generated>\"",
            "path = C:\\Users\\<you>\\project",
            "cd /home/runner/work",
            "password = os.environ['X']",
            "api_key=" + "Zx9Qw8Er7Ty6Ui5" + "  # audit:allow",
        ]
        for line in clean_lines:
            self.assertEqual(list(audit_repo.scan_text(line)), [], line)

    def test_env_file_and_weights_in_tree_are_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / ".env.example").write_text("A=1\n", encoding="utf-8")
            (root / "prod.env").write_text("A=1\n", encoding="utf-8")
            (root / "w.safetensors").write_bytes(b"x")
            (root / "ok.py").write_text("print('hi')\n", encoding="utf-8")
            findings, files, mode = audit_repo.scan(root)
            flagged = {f[0] for f in findings}
            self.assertEqual(flagged, {"prod.env", "w.safetensors"})
            self.assertEqual(mode, "walk")

    def test_repository_itself_is_clean(self):
        findings, _, _ = audit_repo.scan()
        self.assertEqual(findings, [])


class _FakeOllama(http.server.BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *args):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"models": [{"name": "phi3.5:latest"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeOllama.calls.append(req)
        self._send({"message": {"content": json.dumps(
            {"intent": "alarm_set", "date": None, "time": None, "place": None})}})


class LabelLocalTests(unittest.TestCase):
    """Runs scripts/label_local.py against a FAKE Ollama server (not a real teacher model)."""

    def setUp(self):
        _FakeOllama.calls = []
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _FakeOllama)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def run_label(self, *extra):
        argv = ["label_local.py", "--ollama", self.url, "--out-dir", str(Path(self.tmp.name) / "teacher"),
                *extra]
        with mock.patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
            label_local.main()

    def shard(self, i=0):
        p = Path(self.tmp.name) / "teacher" / f"shard-{i}.jsonl"
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]

    def test_labels_limit_and_resumes_without_duplicates(self):
        self.run_label("--limit", "5")
        self.assertEqual(len(self.shard()), 5)
        self.assertEqual(len(_FakeOllama.calls), 5)
        self.run_label("--limit", "3")
        rows = self.shard()
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({r["id"] for r in rows}), 8)
        self.assertEqual(set(rows[0]), {"id", "text", "teacher_raw"})

    def test_request_matches_worker_format(self):
        self.run_label("--limit", "1")
        req = _FakeOllama.calls[0]
        self.assertEqual(req["format"], "json")
        self.assertEqual(req["options"], {"temperature": 0, "seed": 42, "num_predict": 128})
        self.assertEqual(req["model"], "phi3.5")

    def test_sharding_matches_cluster_worker(self):
        self.run_label("--shard-index", "1", "--shard-count", "2", "--limit", "3")
        self.assertEqual([r["id"] for r in self.shard(1)], ["C0002", "C0004", "C0006"])

    def test_unreachable_ollama_gives_friendly_exit(self):
        argv = ["label_local.py", "--ollama", "http://127.0.0.1:9", "--out-dir", self.tmp.name]
        with mock.patch("sys.argv", argv), self.assertRaises(SystemExit) as cm:
            label_local.main()
        self.assertIn("cannot reach Ollama", str(cm.exception))

    def test_model_available(self):
        self.assertTrue(label_local.model_available(["phi3.5:latest"], "phi3.5"))
        self.assertTrue(label_local.model_available(["phi3.5:latest"], "phi3.5:latest"))
        self.assertFalse(label_local.model_available(["llama3:latest"], "phi3.5"))
        self.assertFalse(label_local.model_available(["phi3.5:latest"], "phi3.5:q4"))


class SmokeDatasetTests(unittest.TestCase):
    def test_records_pass_the_real_cleaning_rules(self):
        recs = make_smoke_dataset.build(64)
        self.assertEqual(len(recs), 64)
        kept, stats = clean(recs)
        self.assertEqual(stats["kept"], 64, stats)

    def test_all_eight_intents_covered_and_deterministic(self):
        a, b = make_smoke_dataset.build(40), make_smoke_dataset.build(40)
        self.assertEqual(a, b)
        self.assertEqual(len({json.loads(r["teacher_raw"])["intent"] for r in a}), 8)


if __name__ == "__main__":
    unittest.main()
