import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from local_model_dock.artifact_verification import verify_artifact


class ArtifactVerificationTests(unittest.TestCase):
    def test_valid_artifact_and_modified_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = b'{"model_type":"qwen3_moe"}'
            weights = b"weights"
            (root / "config.json").write_bytes(config)
            (root / "model.safetensors").write_bytes(weights)
            manifest = {
                "version": 1, "repo": "owner/model", "revision": "a" * 40,
                "files": [
                    {"path": "config.json", "size": len(config), "digest_type": "git-blob-sha1",
                     "digest": hashlib.sha1(b"blob " + str(len(config)).encode() + b"\0" + config).hexdigest()},
                    {"path": "model.safetensors", "size": len(weights), "digest_type": "sha256",
                     "digest": hashlib.sha256(weights).hexdigest()},
                ],
            }
            (root / ".local-model-dock-manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_artifact(root)["revision"], "a" * 40)
            (root / "unexpected").write_text("x")
            with self.assertRaises(ValueError):
                verify_artifact(root)
            (root / "unexpected").unlink()
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify_artifact(root)

    def test_invalid_config_json_is_rejected_even_with_matching_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for name, data in (("config.json", b"not json"), ("model.safetensors", b"weights")):
                (root / name).write_bytes(data)
                entries.append({"path": name, "size": len(data), "digest_type": "sha256",
                                "digest": hashlib.sha256(data).hexdigest()})
            (root / ".local-model-dock-manifest.json").write_text(json.dumps({
                "version": 1, "repo": "owner/model", "revision": "a" * 40, "files": entries}))
            with self.assertRaises(ValueError):
                verify_artifact(root)
