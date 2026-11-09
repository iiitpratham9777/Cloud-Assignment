import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from service import settings


class DotenvTests(unittest.TestCase):
    def test_parse_handles_quotes_comments_export(self):
        text = "\n".join([
            "# comment", "A=1", "B = two words ", 'C="quoted # not comment"', "D='single'",
            "export E=5", "F=6 # inline", "NOEQUALS", "=novalue", "G=",
        ])
        got = settings.parse_dotenv(text)
        self.assertEqual(got["A"], "1")
        self.assertEqual(got["B"], "two words")
        self.assertEqual(got["C"], "quoted # not comment")
        self.assertEqual(got["D"], "single")
        self.assertEqual(got["E"], "5")
        self.assertEqual(got["F"], "6")
        self.assertEqual(got["G"], "")
        self.assertNotIn("NOEQUALS", got)

    def test_load_does_not_override_and_skips_blank(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "x.env"
            f.write_text("KEEP=from_file\nNEW=from_file\nBLANK=\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"KEEP": "from_env"}, clear=False):
                os.environ.pop("NEW", None)
                os.environ.pop("BLANK", None)
                applied = settings.load_dotenv(f)
                self.assertEqual(os.environ["KEEP"], "from_env")
                self.assertEqual(os.environ["NEW"], "from_file")
                self.assertNotIn("BLANK", os.environ)
                self.assertEqual(applied, {"NEW": "from_file"})
                os.environ.pop("NEW", None)

    def test_missing_file_is_fine(self):
        self.assertEqual(settings.load_dotenv(Path(tempfile.gettempdir()) / "definitely-missing.env"), {})


class PathTests(unittest.TestCase):
    def test_relative_paths_resolve_against_repo_root_not_cwd(self):
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)
            try:
                self.assertEqual(settings.resolve_path("data/x.jsonl"),
                                 settings.REPO_ROOT / "data" / "x.jsonl")
            finally:
                os.chdir(old)

    def test_absolute_paths_are_kept(self):
        p = Path(tempfile.gettempdir()).resolve() / "abs"
        self.assertEqual(settings.resolve_path(p), p)

    def test_model_dir_default_and_override(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODEL_DIR", None)
            self.assertEqual(settings.model_dir(), settings.REPO_ROOT / "model_out" / "v1")
            os.environ["MODEL_DIR"] = "somewhere/else"
            self.assertEqual(settings.model_dir(), settings.REPO_ROOT / "somewhere" / "else")

    def test_cache_dir_is_portable_temp_dir(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODEL_CACHE_DIR", None)
            self.assertEqual(settings.model_cache_dir().parent, Path(tempfile.gettempdir()))

    def test_env_int(self):
        with mock.patch.dict(os.environ, {"N": "7", "BAD": "x"}, clear=False):
            self.assertEqual(settings.env_int("N", 1), 7)
            self.assertEqual(settings.env_int("UNSET_FOR_TEST", 3), 3)
            with self.assertRaises(SystemExit):
                settings.env_int("BAD", 1)


if __name__ == "__main__":
    unittest.main()
