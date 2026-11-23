import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from local_model_dock.ollama_import import source_gguf


class SourceGGUFTests(unittest.TestCase):
    def test_source_identity_requires_one_verified_gguf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            revision = "a" * 40
            name = "owner--model--" + revision
            model = root / "models" / name
            model.mkdir(parents=True)
            (root / "models").chmod(0o700)
            model.chmod(0o700)
            (model / "weights.gguf").write_bytes(b"weights")
            manifest = {"version": 1, "repo": "owner/model", "revision": revision,
                        "files": [{"path": "weights.gguf", "size": 7,
                                   "digest_type": "sha256",
                                   "digest": hashlib.sha256(b"weights").hexdigest()}]}
            (model / ".local-model-dock-manifest.json").write_text(json.dumps(manifest))
            identity = source_gguf(root, name)
            self.assertEqual(identity["sha256"], hashlib.sha256(b"weights").hexdigest())
            self.assertEqual(identity["repo"], "owner/model")
            (model / "weights.gguf").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                source_gguf(root, name)

    def test_symlinked_model_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir(mode=0o700)
            (root / "models" / "alias").symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                source_gguf(root, "alias")
