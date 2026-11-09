"""API behaviour in local mode (no S3, no Kubernetes)."""
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("SKIP_MODEL_LOAD", "1")

try:
    from fastapi.testclient import TestClient
    from service import app as app_module
    HAVE_FASTAPI = True
except ImportError:
    HAVE_FASTAPI = False

GOOD = '{"intent": "alarm_set", "date": "tomorrow", "time": "seven am", "place": null}'


def reset_state():
    app_module.state.update(generator=None, ready=False, error=None, detail=None)


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx not installed")
class LocalModeTests(unittest.TestCase):
    def setUp(self):
        reset_state()
        self.client = TestClient(app_module.app)
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        reset_state()
        self.tmp.cleanup()

    def test_missing_weights_gives_503_with_actionable_detail(self):
        env = {"MODEL_DIR": str(Path(self.tmp.name) / "nope")}
        with mock.patch.dict(os.environ, env):
            os.environ.pop("MODEL_URI", None)
            app_module._load()
        r = self.client.get("/readyz")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["status"], "loading")
        self.assertIn("Model directory not found", r.json()["detail"])
        self.assertIn("fetch_model.py", r.json()["detail"])
        self.assertEqual(self.client.get("/healthz").status_code, 200)  # process stays alive
        p = self.client.post("/parse", json={"text": "x"})
        self.assertEqual((p.status_code, p.json()), (503, {"error": "model_not_ready"}))

    def test_model_uri_without_endpoint_is_reported(self):
        with mock.patch.dict(os.environ, {"MODEL_URI": "s3://models/v1/"}):
            os.environ.pop("S3_ENDPOINT", None)
            app_module._load()
        self.assertIn("S3_ENDPOINT", self.client.get("/readyz").json()["detail"])

    def test_unexpected_load_error_does_not_leak_details(self):
        d = Path(self.tmp.name)
        for n in ("config.json", "model.safetensors", "tokenizer.json"):
            (d / n).write_bytes(b"x")
        with mock.patch.dict(os.environ, {"MODEL_DIR": str(d)}), \
                mock.patch("service.model_loader.Generator", side_effect=RuntimeError("secret internals")):
            os.environ.pop("MODEL_URI", None)
            app_module._load()
        detail = self.client.get("/readyz").json()["detail"]
        self.assertNotIn("secret internals", detail)
        self.assertIn("server log", detail)

    def test_local_model_dir_is_loaded_and_served(self):
        d = Path(self.tmp.name)
        for n in ("config.json", "model.safetensors", "tokenizer.json"):
            (d / n).write_bytes(b"x")
        seen = {}

        def fake_generator(path, max_new_tokens):
            seen["path"], seen["max"] = Path(path), max_new_tokens
            return lambda text: GOOD

        with mock.patch.dict(os.environ, {"MODEL_DIR": str(d), "MAX_NEW_TOKENS": "17"}), \
                mock.patch("service.model_loader.Generator", side_effect=fake_generator):
            os.environ.pop("MODEL_URI", None)
            app_module._load()
        self.assertEqual(seen, {"path": d, "max": 17})
        self.assertEqual(self.client.get("/readyz").status_code, 200)
        r = self.client.post("/parse", json={"text": "wake me at seven am tomorrow"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["intent"], "alarm_set")

    def test_served_by_falls_back_to_hostname_when_env_missing(self):
        app_module.state.update(ready=True, generator=lambda t: GOOD)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HOSTNAME", None)
            r = self.client.post("/parse", json={"text": "x"})
        self.assertEqual(r.headers["X-Served-By"], socket.gethostname())


class DevScriptTests(unittest.TestCase):
    def test_serve_refuses_to_start_without_weights_with_clear_message(self):
        from scripts import dev
        args = mock.Mock(host=None, port=None, reload=False, allow_no_model=False)
        with mock.patch.dict(os.environ, {"MODEL_DIR": str(Path(tempfile.gettempdir()) / "no-model-here")}):
            os.environ.pop("MODEL_URI", None)
            with self.assertRaises(SystemExit) as cm:
                dev.cmd_serve(args)
        self.assertIn("cannot start", str(cm.exception))
        self.assertIn("fetch_model.py", str(cm.exception))

    def test_venv_python_falls_back_to_current_interpreter(self):
        from scripts import dev
        with mock.patch.object(dev, "VENV", Path(tempfile.gettempdir()) / "no-venv-here"):
            import sys
            self.assertEqual(dev.venv_python(), sys.executable)

    def test_venv_missing_checks_the_interpreter_that_will_run_the_server(self):
        from scripts import dev
        self.assertEqual(dev.venv_missing(["json", "os"]), [])
        self.assertEqual(dev.venv_missing(["json", "surely_not_installed_pkg_xyz"]),
                         ["surely_not_installed_pkg_xyz"])

    def test_cli_parses_every_subcommand(self):
        from scripts import dev
        with mock.patch.object(dev, "cmd_check") as m:
            dev.main(["check", "serve"])
            m.assert_called_once()
        # flags must reach the underlying tool untouched (regression: argparse REMAINDER ate them)
        for cmd, fn in (("train", "cmd_train"), ("test", "cmd_test"), ("audit", "cmd_audit")):
            with self.subTest(cmd=cmd), mock.patch.object(dev, fn) as m:
                with mock.patch.dict(dev.main.__globals__, {fn: m}):
                    dev.main([cmd, "--version", "v2", "-k", "x", "--subset", "5"])
                self.assertEqual(m.call_args[0][0].args, ["--version", "v2", "-k", "x", "--subset", "5"])


if __name__ == "__main__":
    unittest.main()
