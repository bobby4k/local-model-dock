import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from local_model_dock.probe_records import _ollama_layer


class OllamaLayerTests(unittest.TestCase):
    def test_config_blob_must_match_manifest_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            manifest_dir = models / "manifests-v2" / "ollama.com" / "library" / "model"
            manifest_dir.mkdir(parents=True)
            blobs = models / "blobs"
            blobs.mkdir()
            digest = hashlib.sha256(b"weights").hexdigest()
            (blobs / ("sha256-" + digest)).write_bytes(b"weights")
            config = hashlib.sha256(b"config").hexdigest()
            (blobs / ("sha256-" + config)).write_bytes(b"config")
            (manifest_dir / "latest").write_text(json.dumps({
                "config": {"digest": "sha256:" + config, "size": 6},
                "layers": [{"mediaType": "application/vnd.ollama.image.model",
                            "digest": "sha256:" + digest, "size": 7}]}))
            self.assertEqual(_ollama_layer(root, "model"), (digest, 7))
            (blobs / ("sha256-" + config)).write_bytes(b"config!")
            with self.assertRaises(ValueError):
                _ollama_layer(root, "model")

    def test_content_addressed_manifest_symlink_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            manifest_dir = models / "manifests-v2" / "ollama.com" / "library" / "model"
            manifest_dir.mkdir(parents=True)
            blobs = models / "blobs"
            blobs.mkdir()
            data = b"weights"
            digest = hashlib.sha256(data).hexdigest()
            (blobs / ("sha256-" + digest)).write_bytes(data)
            raw = json.dumps({"layers": [{"mediaType": "application/vnd.ollama.image.model",
                                          "digest": "sha256:" + digest, "size": len(data)}]}).encode()
            manifest = blobs / ("sha256-" + hashlib.sha256(raw).hexdigest())
            manifest.write_bytes(raw)
            (manifest_dir / "latest").symlink_to(manifest)
            self.assertEqual(_ollama_layer(root, "model"), (digest, len(data)))
            manifest.write_bytes(raw + b" ")
            with self.assertRaises(ValueError):
                _ollama_layer(root, "model")

    def test_layer_digest_is_checked_against_local_blob(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            manifest_dir = models / "manifests-v2" / "ollama.com" / "library" / "local-dock-initial"
            manifest_dir.mkdir(parents=True)
            blobs = models / "blobs"
            blobs.mkdir()
            content = b"model weights"
            digest = hashlib.sha256(content).hexdigest()
            blob = blobs / ("sha256-" + digest)
            blob.write_bytes(content)
            manifest = {"layers": [{"mediaType": "application/vnd.ollama.image.model",
                                    "digest": "sha256:" + digest, "size": len(content)}]}
            (manifest_dir / "latest").write_text(json.dumps(manifest))
            self.assertEqual(_ollama_layer(root, "local-dock-initial"), (digest, len(content)))
            blob.write_bytes(b"other content!")
            with self.assertRaises(ValueError):
                _ollama_layer(root, "local-dock-initial")

    def test_rejects_ambiguous_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_dir = root / "models" / "manifests-v2" / "ollama.com" / "library" / "model"
            manifest_dir.mkdir(parents=True)
            layer = {"mediaType": "application/vnd.ollama.image.model",
                     "digest": "sha256:" + "a" * 64, "size": 1}
            (manifest_dir / "latest").write_text(json.dumps({"layers": [layer, layer]}))
            with self.assertRaises(ValueError):
                _ollama_layer(root, "model")
