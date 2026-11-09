import json
import tempfile
import unittest
from pathlib import Path

from service import checks
from service.settings import REPO_ROOT


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class PythonVersionTests(unittest.TestCase):
    def test_supported_and_unsupported(self):
        self.assertEqual(checks.check_python((3, 11)), [])
        self.assertEqual(checks.check_python((3, 10)), [])
        self.assertEqual(checks.check_python((3, 12)), [])
        self.assertTrue(checks.check_python((3, 9)))
        self.assertTrue(checks.check_python((3, 13)))

    def test_missing_modules(self):
        self.assertEqual(checks.missing_modules(["json", "os"]), [])
        self.assertEqual(checks.missing_modules(["json", "surely_not_installed_pkg_xyz"]),
                         ["surely_not_installed_pkg_xyz"])


class DatasetCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_train_file_explains_how_to_get_one(self):
        msg = checks.check_train_file(self.dir / "train.jsonl")[0]
        self.assertIn("not found", msg)
        self.assertIn("label_local.py", msg)
        self.assertIn("make_smoke_dataset.py", msg)

    def test_empty_and_malformed(self):
        (self.dir / "e.jsonl").write_text("\n", encoding="utf-8")
        self.assertIn("empty", checks.check_train_file(self.dir / "e.jsonl")[0])
        (self.dir / "b.jsonl").write_text("{not json}\n", encoding="utf-8")
        self.assertIn("not valid JSON", checks.check_train_file(self.dir / "b.jsonl")[0])
        write_jsonl(self.dir / "k.jsonl", [{"id": "1", "text": "x"}])
        self.assertIn("teacher_raw", checks.check_train_file(self.dir / "k.jsonl")[0])

    def test_good_train_file(self):
        write_jsonl(self.dir / "t.jsonl", [{"id": "1", "text": "x", "teacher_raw": "{}"}])
        self.assertEqual(checks.check_train_file(self.dir / "t.jsonl"), [])
        checks.require_train_file(self.dir / "t.jsonl")

    def test_require_raises_user_facing_error(self):
        with self.assertRaises(checks.MissingArtifactError):
            checks.require_train_file(self.dir / "nope.jsonl")

    def test_repo_data_present_and_detected_when_missing(self):
        self.assertEqual(checks.check_repo_data(REPO_ROOT), [])
        problems = checks.check_repo_data(self.dir)
        self.assertEqual(len(problems), 3)


class ModelDirCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "m"

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_dir(self):
        msg = checks.check_model_dir(self.dir)[0]
        self.assertIn("not found", msg)
        self.assertIn("fetch_model.py", msg)

    def test_incomplete_dir_lists_what_is_missing(self):
        self.dir.mkdir()
        (self.dir / "config.json").write_text("{}", encoding="utf-8")
        msg = checks.check_model_dir(self.dir)[0]
        self.assertIn("weights", msg)
        self.assertIn("tokenizer", msg)
        self.assertNotIn("config.json,", msg)

    def test_complete_dir(self):
        self.dir.mkdir()
        for n in ("config.json", "model.safetensors", "tokenizer.json"):
            (self.dir / n).write_bytes(b"x")
        self.assertEqual(checks.check_model_dir(self.dir), [])
        checks.require_model_dir(self.dir)

    def test_pytorch_bin_weights_accepted(self):
        self.dir.mkdir()
        for n in ("config.json", "pytorch_model.bin", "tokenizer_config.json"):
            (self.dir / n).write_bytes(b"x")
        self.assertEqual(checks.check_model_dir(self.dir), [])


if __name__ == "__main__":
    unittest.main()
