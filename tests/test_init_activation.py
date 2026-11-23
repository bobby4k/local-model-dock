import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ActivationTests(unittest.TestCase):
    def run_wrapper(self, shell, wrapper, installer_status=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy(ROOT / "scripts" / wrapper, scripts / wrapper)
            (scripts / "init.sh").write_text(f"#!/bin/sh\nexit {installer_status}\n")
            venv = root / ".venv" / "bin"
            venv.mkdir(parents=True)
            (venv / "activate").write_text("export TEST_ACTIVATED=yes\n")
            (venv / "activate.fish").write_text("set -gx TEST_ACTIVATED yes\n")
            if shell == "fish":
                command = f'source "{scripts / wrapper}"; set result $status; echo "$result:$TEST_ACTIVATED"'
            else:
                command = f'source "{scripts / wrapper}"; result=$?; echo "$result:${{TEST_ACTIVATED:-}}"'
            return subprocess.run([shell, "-c", command], cwd=root, text=True, capture_output=True)

    def test_bash_activates_calling_shell(self):
        result = self.run_wrapper("bash", "init-env.sh")
        self.assertEqual(result.stdout.strip(), "0:yes")

    def test_bash_does_not_activate_after_failure(self):
        result = self.run_wrapper("bash", "init-env.sh", 7)
        self.assertEqual(result.stdout.strip(), "7:")

    @unittest.skipUnless(shutil.which("zsh"), "zsh is not installed")
    def test_zsh_activates_calling_shell(self):
        result = self.run_wrapper("zsh", "init-env.sh")
        self.assertEqual(result.stdout.strip(), "0:yes")

    @unittest.skipUnless(shutil.which("fish"), "fish is not installed")
    def test_fish_activates_calling_shell(self):
        result = self.run_wrapper("fish", "init.fish")
        self.assertEqual(result.stdout.strip(), "0:yes")

    @unittest.skipUnless(shutil.which("fish"), "fish is not installed")
    def test_fish_does_not_activate_after_failure(self):
        result = self.run_wrapper("fish", "init.fish", 7)
        self.assertEqual(result.stdout.strip(), "7:")
