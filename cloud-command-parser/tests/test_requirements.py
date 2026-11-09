"""The Docker image, CI and local installs use different requirement files; keep their pins equal."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([0-9][^\s;]*)")


def pins(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = PIN.match(line.strip())
        if m:
            name = m.group(1).lower().replace("_", "-")
            out.setdefault(name, set()).add(m.group(2).split("+")[0])  # ignore the "+cpu" local label
    return out


class RequirementsConsistencyTests(unittest.TestCase):
    def test_docker_pins_match_requirements_dir(self):
        docker = pins(ROOT / "service" / "requirements.txt")
        merged = {}
        for f in ("api", "ml", "s3"):
            for k, v in pins(ROOT / "requirements" / f"{f}.txt").items():
                merged.setdefault(k, set()).update(v)
        for name, versions in docker.items():
            self.assertIn(name, merged, f"{name} is in service/requirements.txt but not in requirements/")
            self.assertEqual(versions, merged[name], f"version mismatch for {name}")

    def test_dev_pins_match_requirements_dir(self):
        merged = {}
        for f in ("api", "ml", "s3"):
            for k, v in pins(ROOT / "requirements" / f"{f}.txt").items():
                merged.setdefault(k, set()).update(v)
        for name, versions in pins(ROOT / "requirements-dev.txt").items():
            if name in merged:
                self.assertEqual(versions, merged[name], f"version mismatch for {name}")

    def test_macos_has_a_torch_pin_without_cpu_label(self):
        text = (ROOT / "requirements" / "ml.txt").read_text(encoding="utf-8")
        self.assertRegex(text, r'torch==2\.4\.1 ; sys_platform == "darwin"')
        self.assertRegex(text, r'torch==2\.4\.1\+cpu ; sys_platform != "darwin"')

    def test_training_config_library_versions_match_pins(self):
        import yaml
        cfg = yaml.safe_load((ROOT / "training" / "config.yaml").read_text(encoding="utf-8"))
        ml = pins(ROOT / "requirements" / "ml.txt")
        for name in ("torch", "transformers", "tokenizers", "numpy"):
            self.assertEqual({cfg["library_versions"][name]}, ml[name])


if __name__ == "__main__":
    unittest.main()
