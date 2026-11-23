import unittest
import subprocess
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from local_model_dock.hardware import detect_backend, available_memory, available_disk, print_hardware


class HardwareTests(unittest.TestCase):
    def test_macos_memory_display_identifies_immediately_free_pages(self):
        output = StringIO()
        with patch("platform.system", return_value="Darwin"), \
             patch("local_model_dock.hardware.total_memory", return_value=16 * 1024**3), \
             patch("local_model_dock.hardware.available_memory", return_value=1024**3), \
             redirect_stdout(output):
            print_hardware()
        self.assertIn("1.0 GiB immediately free", output.getvalue())
        self.assertNotIn("1.0 GiB available", output.getvalue())

    def test_linux_memory_display_remains_available(self):
        output = StringIO()
        with patch("platform.system", return_value="Linux"), \
             patch("local_model_dock.hardware.available_memory", return_value=1024**3), \
             redirect_stdout(output):
            print_hardware()
        self.assertIn("1.0 GiB available", output.getvalue())

    def test_apple_silicon_uses_ollama(self):
        self.assertEqual(detect_backend("Darwin", "arm64"), "ollama")

    def test_intel_mac_is_unsupported(self):
        with self.assertRaises(ValueError):
            detect_backend("Darwin", "x86_64")

    def test_linux_uses_ollama(self):
        self.assertEqual(detect_backend("Linux", "x86_64"), "ollama")

    def test_unknown_platform_is_unsupported(self):
        with self.assertRaises(ValueError):
            detect_backend("Windows", "AMD64")

    def test_linux_available_memory_from_meminfo(self):
        with patch("pathlib.Path.read_text", return_value="MemTotal: 1000 kB\nMemAvailable: 400 kB\n"):
            self.assertEqual(available_memory("Linux"), 409600)

    def test_unknown_memory_is_not_assumed(self):
        with patch("pathlib.Path.read_text", side_effect=OSError("unavailable")):
            self.assertIsNone(available_memory("Linux"))

    def test_macos_available_memory_only_counts_free_pages(self):
        output = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                  "Pages free: 100.\nPages inactive: 500.\nPages speculative: 20.\n")
        with patch("subprocess.check_output", return_value=output):
            self.assertEqual(available_memory("Darwin"), 100 * 16384)

    def test_macos_invalid_or_missing_vm_statistics_are_unknown(self):
        outputs = [
            "", "Mach Virtual Memory Statistics: (page size of 0 bytes)\nPages free: 5.\n",
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n",
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: -1.\n",
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 2\n",
        ]
        for output in outputs:
            with self.subTest(output=output), patch("subprocess.check_output", return_value=output):
                self.assertIsNone(available_memory("Darwin"))
        with patch("subprocess.check_output", side_effect=OSError):
            self.assertIsNone(available_memory("Darwin"))
        with patch("subprocess.check_output", side_effect=subprocess.TimeoutExpired("vm_stat", 2)):
            self.assertIsNone(available_memory("Darwin"))

    def test_macos_rejects_free_memory_above_physical_memory(self):
        output = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 100.\n"
        for total in (16384, None):
            with self.subTest(total=total), patch("subprocess.check_output", return_value=output), \
                 patch("local_model_dock.hardware.total_memory", return_value=total):
                self.assertIsNone(available_memory("Darwin"))

    def test_disk_space_uses_requested_storage_path_and_fails_closed(self):
        with patch("shutil.disk_usage") as usage:
            usage.return_value.free = 123
            self.assertEqual(available_disk("/models"), 123)
            usage.assert_called_once_with("/models")
        with patch("shutil.disk_usage", side_effect=OSError):
            self.assertIsNone(available_disk("/missing"))
