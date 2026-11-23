import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "init.sh"


class InitScriptTests(unittest.TestCase):
    def run_init(self, system, machine, installed=False, apt_candidate=True, existing_env=None, brew=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            shutil.copy(SCRIPT, scripts / "init.sh")
            (root / ".env.example").write_text("MODEL_BUDGET_GB=20\n")
            if existing_env is not None:
                (root / ".env").write_text(existing_env)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            log = root / "commands"
            commands = {
                "uname": f'if [ "$1" = "-s" ]; then echo {system}; else echo {machine}; fi',
                "python3": 'mkdir -p .venv/bin; cp "$FAKE_BIN/python" .venv/bin/python',
                "python": f'if [ "$1" = "-c" ]; then exec "{sys.executable}" "$@"; fi; echo "pip $*" >> "$LOG"',
                "sudo": 'echo "sudo $*" >> "$LOG"',
                "apt-cache": f'echo "Candidate: {"1.0" if apt_candidate else "(none)"}"',
            }
            if brew:
                commands["brew"] = 'echo "brew $*" >> "$LOG"'
            if installed:
                commands["ollama"] = 'exit 0'
            for name, body in commands.items():
                path = bin_dir / name
                path.write_text("#!/bin/sh\n" + body + "\n")
                path.chmod(0o755)
            env = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin",
                   "FAKE_BIN": str(bin_dir), "LOG": str(log)}
            result = subprocess.run(["bash", str(scripts / "init.sh")], cwd=root,
                                    env=env, capture_output=True, text=True)
            env_file = root / ".env"
            return result, log.read_text() if log.exists() else "", env_file.read_text() if env_file.exists() else None

    def test_mac_installs_ollama_and_project(self):
        result, log, env_file = self.run_init("Darwin", "arm64")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("brew install ollama\n", log)
        self.assertNotIn("brew services", log)
        self.assertIn("pip -m pip install -e .", log)
        self.assertRegex(env_file, r"(?m)^API_KEY=[0-9a-f]{64}$")

    def test_existing_backend_is_not_reinstalled(self):
        result, log, env_file = self.run_init("Darwin", "arm64", installed=True, brew=False, existing_env="CUSTOM=1\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("brew install", log)
        self.assertTrue(env_file.startswith("CUSTOM=1\n"))
        self.assertRegex(env_file, r"(?m)^API_KEY=[0-9a-f]{64}$")

    def test_missing_homebrew_fails_without_installing(self):
        result, log, _ = self.run_init("Darwin", "arm64", brew=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Homebrew", result.stderr)
        self.assertEqual(log, "")

    def test_existing_linux_backend_is_not_reinstalled(self):
        result, log, _ = self.run_init("Linux", "x86_64", installed=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("apt-get install", log)

    def test_existing_key_is_preserved(self):
        result, _, env_file = self.run_init(
            "Darwin", "arm64", installed=True, existing_env="API_KEY=my-secret\nCUSTOM=1\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(env_file, "API_KEY=my-secret\nCUSTOM=1\n")

    def test_empty_key_is_generated(self):
        result, _, env_file = self.run_init(
            "Darwin", "arm64", installed=True, existing_env='API_KEY=""\n'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(env_file, r"(?m)^API_KEY=[0-9a-f]{64}$")

    def test_linux_uses_apt(self):
        result, log, _ = self.run_init("Linux", "x86_64")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sudo apt-get install -y ollama", log)

    def test_intel_mac_rejected_before_install(self):
        result, log, _ = self.run_init("Darwin", "x86_64")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(log, "")

    def test_missing_apt_package_fails_without_installing(self):
        result, log, _ = self.run_init("Linux", "x86_64", apt_candidate=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unavailable", result.stderr)
        self.assertNotIn("apt-get install", log)
