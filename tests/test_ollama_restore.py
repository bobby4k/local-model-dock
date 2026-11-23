import tempfile
import unittest
import sys
from io import BytesIO, StringIO
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch, Mock

from local_model_dock.supervisor import ModelState, load_enabled, save_enabled


class RestoreTests(unittest.TestCase):
    def test_start_reports_restored_model_accurately(self):
        from local_model_dock.supervisor import start
        process = Mock()
        process.stdout = BytesIO(b"READY ENABLED\n")
        output = StringIO()
        with patch("local_model_dock.supervisor.load_config", return_value=Mock(api_key="secret")), \
             patch("local_model_dock.supervisor.private_root"), \
             patch("local_model_dock.supervisor.subprocess.Popen", return_value=process), \
             patch("local_model_dock.supervisor.select.select", return_value=([process.stdout], [], [])), \
             redirect_stdout(output):
            start()
        self.assertIn("restored", output.getvalue().lower())

    def test_cli_verifies_and_enables_authorized_model_without_discovery(self):
        from local_model_dock.cli import main
        with patch.object(sys, "argv", ["dock", "update", "--verify", "local-dock-initial", "--activate"]), \
             patch("local_model_dock.cli._control", return_value={"verified": True}) as control:
            self.assertEqual(main(), 0)
        self.assertEqual([call.args for call in control.call_args_list],
                         [("verify", "local-dock-initial"), ("check", "local-dock-initial"),
                          ("enable", "local-dock-initial")])

    def test_restart_rechecks_record_before_restoring_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_enabled(root, "local-dock-initial")
            self.assertEqual(load_enabled(root), "local-dock-initial")
            with patch("local_model_dock.supervisor.check_record", return_value=False):
                state = ModelState(object())
                self.assertFalse(state.enable(load_enabled(root)))
                self.assertIsNone(state.model)
            save_enabled(root, None)
            self.assertIsNone(load_enabled(root))

    def test_invalid_or_symlinked_selection_is_not_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "enabled-model").write_text("../other")
            self.assertIsNone(load_enabled(root))
            (root / "enabled-model").unlink()
            (root / "enabled-model").symlink_to(root / "other")
            self.assertIsNone(load_enabled(root))
