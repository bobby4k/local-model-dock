import tempfile
import unittest
import shutil
import os
import socket
from pathlib import Path
from unittest.mock import patch

from local_model_dock.storage import Confirmation, download, inspect_storage


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.confirmation = Confirmation('model', 'revision')

    def run_download(self, **kwargs):
        args = dict(root=self.root, name='model', revision='revision', size_bytes=4,
                    budget_bytes=100, cache_bytes=0, confirmation=self.confirmation,
                    write=lambda stream: stream.write(b'data'))
        args.update(kwargs)
        return download(**args)

    def test_confirmation_is_bound_to_candidate_and_revision(self):
        for confirmation in (None, Confirmation('other', 'revision'), Confirmation('model', 'old')):
            with self.subTest(confirmation=confirmation), self.assertRaises(PermissionError):
                self.run_download(confirmation=confirmation)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_unsafe_paths_and_symlinks_are_rejected(self):
        for name in ('../outside', '/tmp/outside', '.', 'a/b'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.run_download(name=name, confirmation=Confirmation(name, 'revision'))
        (self.root / 'link').symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.run_download()

    def test_insufficient_space_does_not_download(self):
        usage = shutil.disk_usage(self.root)._replace(free=2)
        with patch('local_model_dock.storage.shutil.disk_usage', return_value=usage):
            with self.assertRaises(OSError):
                self.run_download()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_failed_download_leaves_existing_files_and_no_temporary_file(self):
        (self.root / 'current').write_bytes(b'current')
        def fail(stream):
            stream.write(b'bad')
            raise OSError('network')
        with self.assertRaises(OSError):
            self.run_download(write=fail)
        self.assertEqual((self.root / 'current').read_bytes(), b'current')
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['current'])

    def test_budget_warning_and_success(self):
        (self.root / 'current').write_bytes(b'1234')
        result = self.run_download(budget_bytes=5)
        self.assertTrue(result.over_budget)
        self.assertEqual(result.path.read_bytes(), b'data')
        self.assertEqual(result.used_bytes, 8)

    def test_existing_target_and_wrong_size_are_never_replaced(self):
        (self.root / 'model').write_bytes(b'old')
        with self.assertRaises(FileExistsError):
            self.run_download()
        self.assertEqual((self.root / 'model').read_bytes(), b'old')
        with self.assertRaises(ValueError):
            self.run_download(name='another', confirmation=Confirmation('another', 'revision'),
                              write=lambda stream: stream.write(b'x'))
        self.assertFalse((self.root / 'another').exists())

    def test_project_layout_counts_cache_temporary_and_hardlinks_once(self):
        models = self.root / "models"
        models.mkdir()
        (self.root / "cache").mkdir()
        (models / "weights").write_bytes(b"1234")
        (self.root / "cache" / "ssd").write_bytes(b"12345")
        (models / ".download-active").mkdir()
        (models / ".download-active" / "partial").write_bytes(b"123")
        os.link(models / "weights", self.root / "cache" / "link")
        with socket.socket(socket.AF_UNIX) as sock:
            sock.bind(str(self.root / "control.sock"))
            status = inspect_storage(models, 2, 100, 0, project_root=self.root)
        self.assertEqual(status.used_bytes, 12)
        self.assertEqual(status.projected_bytes, 14)

    def test_project_layout_rejects_symlinks_and_double_counted_cache(self):
        models = self.root / "models"
        models.mkdir()
        with self.assertRaises(ValueError):
            inspect_storage(models, 1, 100, 3, project_root=self.root)
        (self.root / "cache").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            inspect_storage(models, 1, 100, 0, project_root=self.root)

    def test_project_layout_rejects_unexpected_special_files(self):
        models = self.root / "models"
        models.mkdir()
        os.mkfifo(self.root / "unknown")
        with self.assertRaises(ValueError):
            inspect_storage(models, 1, 100, 0, project_root=self.root)

    def test_project_layout_requires_direct_models_directory(self):
        with self.assertRaises(ValueError):
            inspect_storage(self.root, 1, 100, 0, project_root=self.root)
