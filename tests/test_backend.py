import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from local_model_dock.backend import Backend, BackendError


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode


class BackendTests(unittest.TestCase):
    def test_ollama_is_isolated_and_stops_only_its_child(self):
        with TemporaryDirectory() as tmp, patch("shutil.which", return_value="/bin/ollama"), \
                patch("subprocess.Popen", return_value=FakeProcess()) as popen:
            backend = Backend("ollama", Path(tmp), port=12345)
            backend.start()
            args, kwargs = popen.call_args
            self.assertEqual(args[0], ["/bin/ollama", "serve"])
            self.assertEqual(kwargs["env"]["OLLAMA_HOST"], "127.0.0.1:12345")
            self.assertEqual(kwargs["env"]["OLLAMA_MODELS"], str(Path(tmp).resolve() / "models"))
            child = backend.process
            backend.stop()
            self.assertTrue(child.terminated)
            self.assertIsNone(backend.process)

    def test_omlx_uses_local_paths_and_explicit_cache_limit(self):
        with TemporaryDirectory() as tmp, patch("shutil.which", return_value="/bin/omlx"), \
                patch("subprocess.Popen", return_value=FakeProcess()) as popen:
            backend = Backend("omlx", Path(tmp), cache_gb=3)
            backend.start()
            command = popen.call_args.args[0]
            self.assertEqual(command[:2], ["/bin/omlx", "serve"])
            for option, value in {
                "--host": "127.0.0.1", "--base-path": str(Path(tmp).resolve()),
                "--model-dir": str(Path(tmp).resolve() / "models"),
                "--paged-ssd-cache-dir": str(Path(tmp).resolve() / "cache"),
                "--paged-ssd-cache-max-size": "3GB",
            }.items():
                self.assertEqual(command[command.index(option) + 1], value)
            self.assertIn("--no-hf-cache", command)
            backend.stop()

    def test_missing_dependency_is_clear(self):
        with TemporaryDirectory() as tmp, patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(BackendError, "ollama.*not found"):
                Backend("ollama", Path(tmp)).start()

    def test_restart_stops_child_before_starting_another(self):
        first, second = FakeProcess(), FakeProcess()
        with TemporaryDirectory() as tmp, patch("shutil.which", return_value="/bin/ollama"), \
                patch("subprocess.Popen", side_effect=[first, second]):
            backend = Backend("ollama", Path(tmp))
            backend.start()
            backend.restart()
            self.assertTrue(first.terminated)
            self.assertIs(backend.process, second)
            backend.stop()

    def test_rejects_symlink_escape_and_invalid_configuration(self):
        with TemporaryDirectory() as tmp, TemporaryDirectory() as outside:
            (Path(tmp) / "models").symlink_to(outside)
            with patch("shutil.which", return_value="/bin/ollama"), \
                    self.assertRaises(BackendError):
                Backend("ollama", Path(tmp)).start()
            for kind, port, cache in [("other", 1, 1), ("ollama", 0, 1), ("omlx", 1, 0)]:
                with self.assertRaises(ValueError):
                    Backend(kind, Path(tmp), port=port, cache_gb=cache)

    def test_timeout_does_not_forget_child_or_start_replacement(self):
        child = FakeProcess()
        child.wait = lambda timeout=None: (_ for _ in ()).throw(subprocess.TimeoutExpired("backend", timeout))
        with TemporaryDirectory() as tmp, patch("shutil.which", return_value="/bin/ollama"), \
                patch("subprocess.Popen", return_value=child) as popen:
            backend = Backend("ollama", Path(tmp))
            backend.start()
            with self.assertRaises(BackendError):
                backend.restart()
            self.assertIs(backend.process, child)
            self.assertEqual(popen.call_count, 1)
