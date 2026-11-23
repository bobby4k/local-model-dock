import os
import subprocess
import sys
import unittest


class CliTests(unittest.TestCase):
    def test_unimplemented_lifecycle_commands_fail_safely(self):
        for command in ("start", "update", "stop", "restart"):
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-m", "local_model_dock", command],
                    capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"},
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("not implemented", result.stderr)
