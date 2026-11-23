import tempfile
import unittest
from pathlib import Path

from local_model_dock.config import load_config


class ConfigTests(unittest.TestCase):
    def test_defaults_and_missing_telegram(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(Path(directory) / ".env", {})
        self.assertEqual(config.budget_gb, 20)
        self.assertTrue(config.auto_download)
        self.assertFalse(config.telegram_enabled)

    def test_environment_overrides_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("MODEL_BUDGET_GB=25\nAUTO_DOWNLOAD=false\n")
            config = load_config(path, {"MODEL_BUDGET_GB": "30"})
        self.assertEqual(config.budget_gb, 30)
        self.assertFalse(config.auto_download)

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            load_config(Path("/nonexistent"), {"MODEL_BUDGET_GB": "-1"})
