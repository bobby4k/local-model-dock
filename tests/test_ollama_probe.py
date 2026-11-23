import hashlib
import http.server
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from local_model_dock.probe_records import verify_model, check_record, _ollama_show


class OllamaProbeTests(unittest.TestCase):
    def test_managed_ollama_does_not_prune_existing_blobs(self):
        from local_model_dock.backend import Backend
        with tempfile.TemporaryDirectory() as root, \
             patch("local_model_dock.backend.shutil.which", return_value="/bin/ollama"), \
             patch("local_model_dock.backend.subprocess.Popen") as popen:
            popen.return_value.poll.return_value = None
            Backend("ollama", Path(root)).start()
            self.assertEqual(popen.call_args.kwargs["env"]["OLLAMA_NOPRUNE"], "1")

    def test_verbose_show_accepts_large_bounded_response(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps({"large": "x" * 1_100_000}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True
        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            self.assertEqual(len(_ollama_show(server.server_port, "model")["large"]), 1_100_000)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        model_dir = self.root / "models" / "manifests-v2" / "ollama.com" / "library" / "local-dock-initial"
        model_dir.mkdir(parents=True)
        (model_dir / "latest").write_text('{"layers":[]}')
        executable = self.root / "ollama"
        executable.write_text("#!/bin/sh\nprintf 'ollama version is 0.40.17\\n'\n")
        executable.chmod(0o700)
        self.backend = SimpleNamespace(kind="ollama", root=self.root, port=12345,
            process=SimpleNamespace(args=[str(executable), "serve"], poll=lambda: None))
        self.backend.restart = lambda: None
        self.show = {"model_info": {"general.parameter_count": 9197093888,
                                    "general.architecture": "qwen35"},
                     "details": {"family": "qwen35"}}
        self.response = {"model": "local-dock-initial:latest", "choices": [{"message": {
            "tool_calls": [{"type": "function", "function": {"name": "lookup",
                           "arguments": '{"query":"dock"}'}}]}}]}
        self.digest = "de3fea7a12b793a573df636eb1c16da3f7960af0377f79e04bfc04a8e65dfb4c"

    @patch("local_model_dock.probe_records.process_owns_loopback_port", return_value=True)
    @patch("local_model_dock.probe_records._ollama_layer")
    @patch("local_model_dock.probe_records._ollama_show")
    @patch("local_model_dock.probe_records._request")
    def test_probe_record_is_bound_to_model_and_backend(self, request, show, layer, owns):
        request.return_value = self.response
        show.return_value = self.show
        layer.return_value = (self.digest, 5780093088)
        self.assertTrue(verify_model(self.backend, "local-dock-initial"))
        self.assertTrue(check_record(self.backend, "local-dock-initial"))
        show.return_value = {**self.show, "model_info": {"general.parameter_count": 1,
                                                         "general.architecture": "qwen35"}}
        self.assertFalse(check_record(self.backend, "local-dock-initial"))

    @patch("local_model_dock.probe_records.process_owns_loopback_port", return_value=True)
    @patch("local_model_dock.probe_records._ollama_layer")
    @patch("local_model_dock.probe_records._ollama_show")
    @patch("local_model_dock.probe_records._request")
    def test_wrong_layer_or_failed_probe_never_produces_record(self, request, show, layer, owns):
        show.return_value = self.show
        layer.return_value = ("0" * 64, 5780093088)
        self.assertFalse(verify_model(self.backend, "local-dock-initial"))
        request.assert_not_called()
        layer.return_value = (self.digest, 5780093088)
        request.return_value = {"model": "local-dock-initial:latest", "choices": []}
        self.assertFalse(verify_model(self.backend, "local-dock-initial"))
        self.assertFalse(check_record(self.backend, "local-dock-initial"))

    @patch("local_model_dock.probe_records.process_owns_loopback_port", return_value=True)
    @patch("local_model_dock.probe_records._ollama_layer")
    @patch("local_model_dock.probe_records._ollama_show")
    @patch("local_model_dock.probe_records._request")
    def test_record_rejected_if_binary_family_or_listener_changes(self, request, show, layer, owns):
        request.return_value = self.response
        show.return_value = self.show
        layer.return_value = (self.digest, 5780093088)
        self.assertTrue(verify_model(self.backend, "local-dock-initial"))
        show.return_value = {**self.show, "details": {"family": "other"}}
        self.assertFalse(check_record(self.backend, "local-dock-initial"))
        show.return_value = self.show
        owns.return_value = False
        self.assertFalse(check_record(self.backend, "local-dock-initial"))
        owns.return_value = True
        with (self.root / "ollama").open("a") as stream:
            stream.write("# changed\n")
        self.assertFalse(check_record(self.backend, "local-dock-initial"))
