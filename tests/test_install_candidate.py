import io
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_model_dock.discovery import DiscoveryCandidate
from local_model_dock.install_candidate import install_candidate
from local_model_dock.storage import Confirmation


class Terminal(io.StringIO):
    def isatty(self):
        return True


class InstallCandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)
        self.root = self.project / "models"
        self.root.mkdir()
        self.rev = "a" * 40
        self.model = DiscoveryCandidate(
            "owner/model", self.rev, "apache-2.0", 10, moe=True,
            formats=("mlx",), accessible=True, disk_bytes=100,
        )
        self.out = Terminal()

    def run_install(self, answers, **options):
        return install_candidate(
            self.model, backend="omlx", root=self.root, cache_bytes=0,
            memory_bytes=10**12, input_stream=answers, output_stream=self.out,
            **options,
        )

    def test_exact_confirmation_downloads_and_displays_risks(self):
        with patch("local_model_dock.install_candidate.model_download.download", return_value=self.root) as download:
            result = self.run_install(Terminal(f"owner/model@{self.rev}\n"))
        self.assertEqual(result, self.root)
        self.assertEqual(download.call_args.kwargs["confirmation"], Confirmation("owner/model", self.rev))
        self.assertIsNone(download.call_args.kwargs["max_confirmed_projected_bytes"])
        for text in ("huggingface.co/owner/model", self.rev, "100", "apache-2.0",
                     "memory", "disk", "tool-call", "not verified"):
            self.assertIn(text, self.out.getvalue())

    def test_generic_or_wrong_confirmation_never_downloads(self):
        for answer in ("yes", "owner/model", f"owner/model@{'b' * 40}"):
            with self.subTest(answer=answer), patch("local_model_dock.install_candidate.model_download.download") as download:
                with self.assertRaises(PermissionError):
                    self.run_install(Terminal(answer + "\n"))
                download.assert_not_called()

    def test_noninteractive_refuses(self):
        with patch("local_model_dock.install_candidate.model_download.download") as download:
            with self.assertRaises(PermissionError):
                self.run_install(io.StringIO(f"owner/model@{self.rev}\n"))
            download.assert_not_called()

    def test_over_budget_requires_separate_exact_confirmation(self):
        with patch("local_model_dock.install_candidate.model_download.download") as download:
            with self.assertRaises(PermissionError):
                self.run_install(Terminal(f"owner/model@{self.rev}\nyes\n"), budget_bytes=50)
            download.assert_not_called()
        with patch("local_model_dock.install_candidate.model_download.download") as download:
            self.run_install(Terminal(f"owner/model@{self.rev}\nover-budget owner/model@{self.rev} 100\n"), budget_bytes=50)
            self.assertEqual(download.call_args.kwargs["max_confirmed_projected_bytes"], 100)

    def test_unknown_size_and_mutable_revision_refused(self):
        from dataclasses import replace
        for model in (replace(self.model, disk_bytes=None), replace(self.model, revision="main")):
            self.model = model
            with self.assertRaises(ValueError):
                self.run_install(Terminal(""))

    def test_recount_after_confirmation_rejects_increased_storage(self):
        from local_model_dock import model_download

        cache = self.project / "cache"
        cache.mkdir()

        def metadata(_url):
            (cache / "other").write_bytes(b"x")
            response = io.BytesIO(json.dumps([{
                "type": "file", "path": "model", "size": 100,
                "lfs": {"oid": "a" * 64},
            }]).encode())
            response.headers = {}
            return response

        with patch.object(model_download, "_open", side_effect=metadata) as network:
            with self.assertRaises(PermissionError):
                self.run_install(Terminal(
                    f"owner/model@{self.rev}\nover-budget owner/model@{self.rev} 100\n"
                ), budget_bytes=50)
        self.assertEqual(network.call_count, 1)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual((cache / "other").read_bytes(), b"x")

    def test_real_layout_counts_sibling_cache_and_passes_project_root(self):
        project = self.project
        (project / "cache").mkdir()
        (project / "cache" / "ssd").write_bytes(b"12345")
        with socket.socket(socket.AF_UNIX) as sock:
            sock.bind(str(project / "control.sock"))
            with patch("local_model_dock.install_candidate.model_download.download") as download:
                self.run_install(Terminal(
                    f"owner/model@{self.rev}\nover-budget owner/model@{self.rev} 105\n"
                ), budget_bytes=100)
        self.assertEqual(download.call_args.kwargs["project_root"], project)
        self.assertEqual(download.call_args.kwargs["max_confirmed_projected_bytes"], 105)


if __name__ == "__main__":
    unittest.main()
