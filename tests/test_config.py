import tempfile
import unittest
from pathlib import Path

from local_model_dock.config import load_config


class ConfigTests(unittest.TestCase):
    def test_fallback_quantization_default_override_and_invalid(self):
        self.assertEqual(load_config(Path("/nonexistent"), {}).fallback_quantization, "i1-Q4_K_M")
        self.assertEqual(load_config(Path("/nonexistent"), {"FALLBACK_QUANTIZATION": "Q8_0"}).fallback_quantization, "Q8_0")
        for value in ("", "../Q4", "nonsense"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "FALLBACK_QUANTIZATION"):
                load_config(Path("/nonexistent"), {"FALLBACK_QUANTIZATION": value})
    def test_defaults_and_missing_telegram(self):
        with tempfile.TemporaryDirectory() as directory:
            config = load_config(Path(directory) / ".env", {})
        self.assertEqual(config.budget_gb, 50)
        self.assertEqual(config.fallback_url, "https://huggingface.co/mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF/tree/main")
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

    def test_fallback_url_is_configurable_and_limited_to_huggingface_repo(self):
        config = load_config(Path("/nonexistent"), {
            "FALLBACK_MODEL_URL": "https://huggingface.co/owner/model/tree/main"})
        self.assertEqual(config.fallback_repo, "owner/model")
        with self.assertRaises(ValueError):
            load_config(Path("/nonexistent"), {"FALLBACK_MODEL_URL": "https://evil.example/owner/model"})
