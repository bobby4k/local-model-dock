import hashlib
import io
import json
import shutil
import tempfile
import unittest
import urllib.request
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from local_model_dock import model_download as md


class Response(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {}


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.revision = "a" * 40
        self.confirmation = md.Confirmation("owner/model", self.revision)
        self.data = b"model"
        self.entries = [{"type": "file", "path": "model.bin", "size": 5,
                         "lfs": {"oid": hashlib.sha256(self.data).hexdigest()}}]

    def run_download(self, **kwargs):
        options = dict(root=self.root, repo="owner/model", revision=self.revision,
                       confirmation=self.confirmation, cache_bytes=0)
        options.update(kwargs)
        return md.download(**options)

    def response(self, url):
        if "/api/models/" in url:
            return Response(json.dumps(self.entries).encode())
        return Response(self.data)

    def test_download_publishes_verified_files(self):
        with patch.object(md, "_open", side_effect=self.response):
            result = self.run_download()
        self.assertEqual((result / "model.bin").read_bytes(), self.data)
        manifest = json.loads((result / ".local-model-dock-manifest.json").read_text())
        self.assertEqual(manifest, {
            "version": 1, "repo": "owner/model", "revision": self.revision,
            "files": [{"path": "model.bin", "size": 5, "digest_type": "sha256",
                       "digest": hashlib.sha256(self.data).hexdigest()}],
        })

    def test_repository_cannot_supply_manifest(self):
        self.entries[0]["path"] = ".local-model-dock-manifest.json"
        with patch.object(md, "_open", side_effect=self.response):
            with self.assertRaises(ValueError):
                self.run_download()

    def test_confirmation_is_required_before_network(self):
        with patch.object(md, "_open") as network:
            with self.assertRaises(PermissionError):
                self.run_download(confirmation=None)
        network.assert_not_called()

    def test_checksum_failure_cleans_only_own_temporary_directory(self):
        (self.root / "keep").write_text("keep")
        self.entries[0]["lfs"]["oid"] = "0" * 64
        with patch.object(md, "_open", side_effect=self.response):
            with self.assertRaises(ValueError):
                self.run_download()
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["keep"])

    def test_non_lfs_file_must_match_git_blob(self):
        self.entries = [{"type": "file", "path": "config.json", "size": 5,
                         "oid": "0" * 40}]
        with patch.object(md, "_open", side_effect=self.response):
            with self.assertRaises(ValueError):
                self.run_download()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_non_lfs_file_accepts_matching_git_blob(self):
        digest = hashlib.sha1(b"blob 5\0" + self.data).hexdigest()
        self.entries = [{"type": "file", "path": "config.json", "size": 5,
                         "oid": digest}]
        with patch.object(md, "_open", side_effect=self.response):
            result = self.run_download()
        self.assertEqual((result / "config.json").read_bytes(), self.data)

    def test_traversal_is_rejected(self):
        self.entries[0]["path"] = "../outside"
        with patch.object(md, "_open", side_effect=self.response):
            with self.assertRaises(ValueError):
                self.run_download()

    def test_budget_requires_separate_confirmation(self):
        with patch.object(md, "_open", side_effect=self.response):
            with self.assertRaises(PermissionError):
                self.run_download(budget_bytes=1)
            result = self.run_download(budget_bytes=1, max_confirmed_projected_bytes=5)
        self.assertTrue(result.is_dir())

    def test_boolean_confirmation_cannot_authorize_increased_usage(self):
        (self.root / "existing").write_bytes(b"12345")
        with patch.object(md, "_open", side_effect=self.response) as network:
            with self.assertRaises((PermissionError, TypeError)):
                self.run_download(budget_bytes=1, confirm_over_budget=True)
        network.assert_not_called()
        self.assertEqual((self.root / "existing").read_bytes(), b"12345")

    def test_projected_usage_above_confirmed_limit_is_refused(self):
        (self.root / "existing").write_bytes(b"12345")
        with patch.object(md, "_open", side_effect=self.response) as network:
            with self.assertRaises(PermissionError):
                self.run_download(budget_bytes=1, max_confirmed_projected_bytes=9)
        self.assertEqual(network.call_count, 1)
        self.assertEqual((self.root / "existing").read_bytes(), b"12345")

    def test_invalid_limits_cannot_authorize_download(self):
        for limit in (True, False, -1, 5.0, "5"):
            with self.subTest(limit=limit), patch.object(md, "_open", side_effect=self.response):
                with self.assertRaises((ValueError, PermissionError)):
                    self.run_download(budget_bytes=1, max_confirmed_projected_bytes=limit)

    def test_confirmed_limit_allows_lower_projected_usage(self):
        with patch.object(md, "_open", side_effect=self.response):
            result = self.run_download(budget_bytes=1, max_confirmed_projected_bytes=6)
        self.assertTrue(result.is_dir())

    def test_redirect_allowlist(self):
        handler = md.SafeRedirect()
        self.assertTrue(handler.allowed("https://cas-bridge.xethub.hf.co/file"))
        for url in ("http://huggingface.co/file", "https://huggingface.co.evil.test/",
                    "https://user@huggingface.co/", "https://127.0.0.1/"):
            self.assertFalse(handler.allowed(url))
            with self.assertRaises(ValueError):
                handler.redirect_request(urllib.request.Request("https://huggingface.co/a"),
                                         None, 302, "redirect", Message(), url)

    def test_existing_model_is_not_overwritten(self):
        target = self.root / ("owner--model--" + self.revision)
        target.mkdir()
        (target / "existing").write_text("original")
        with patch.object(md, "_open", side_effect=self.response):
            with self.assertRaises(FileExistsError):
                self.run_download()
        self.assertEqual((target / "existing").read_text(), "original")

    def test_mutable_revision_is_rejected_before_network(self):
        with patch.object(md, "_open") as network:
            with self.assertRaises(ValueError):
                self.run_download(revision="main", confirmation=md.Confirmation("owner/model", "main"))
        network.assert_not_called()

    def test_size_mismatch_is_rejected(self):
        self.entries[0]["size"] = 4
        with patch.object(md, "_open", side_effect=self.response):
            with self.assertRaises(ValueError):
                self.run_download()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_insufficient_disk_space_prevents_file_download(self):
        with patch.object(md, "_open", side_effect=self.response) as network:
            with patch("local_model_dock.storage.shutil.disk_usage",
                       return_value=shutil.disk_usage(self.root)._replace(free=1)):
                with self.assertRaises(OSError):
                    self.run_download()
        self.assertEqual(network.call_count, 1)
