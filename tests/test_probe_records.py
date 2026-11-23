import hashlib
import http.server
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from local_model_dock.probe_records import verify_model, check_record
from local_model_dock.supervisor import ModelState


class ProbeRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.name = "owner--model--" + "a" * 40
        model = self.root / "models" / self.name
        model.mkdir(parents=True)
        files = []
        for name, data in (("config.json", b"{}"), ("model.safetensors", b"weights")):
            (model / name).write_bytes(data)
            files.append({"path": name, "size": len(data), "digest_type": "sha256",
                          "digest": hashlib.sha256(data).hexdigest()})
        (model / ".local-model-dock-manifest.json").write_text(json.dumps(
            {"version": 1, "repo": "owner/model", "revision": "a" * 40, "files": files}))
        executable = self.root / "omlx"
        executable.write_text("#!/bin/sh\nprintf '1.2.3\\n'\n")
        executable.chmod(0o700)
        command = [str(executable), "serve", "--model-dir", str(self.root / "models"),
                   "--no-hf-cache"]
        self.backend = SimpleNamespace(kind="omlx", root=self.root, port=12345,
                                       process=SimpleNamespace(args=command, pid=123,
                                                               poll=lambda: None))
        self.restarts = []
        self.backend.restart = lambda: self.restarts.append(True)
        limits = patch("local_model_dock.probe_records._model_limits_verified", return_value=True)
        limits.start()
        self.addCleanup(limits.stop)
        self.limits = limits
        self.response = {"model": self.name, "choices": [{"message": {"tool_calls": [
            {"type": "function", "function": {"name": "lookup",
             "arguments": '{"query":"dock"}'}}]}}]}

    @patch("local_model_dock.probe_records.process_owns_loopback_port", return_value=True)
    @patch("local_model_dock.probe_records._request")
    def test_unknown_model_limits_fail_closed(self, request, owns):
        self.limits.stop()
        request.return_value = self.response
        self.assertFalse(verify_model(self.backend, self.name))
        request.assert_not_called()
        self.assertFalse(check_record(self.backend, self.name))

    @patch("local_model_dock.probe_records.process_owns_loopback_port", return_value=True)
    @patch("local_model_dock.probe_records._request")
    def test_legacy_record_cannot_enable_without_parameter_evidence(self, request, owns):
        request.return_value = self.response
        self.assertTrue(verify_model(self.backend, self.name))
        self.limits.stop()
        state = ModelState(self.backend)
        self.assertFalse(state.enable(self.name))
        self.assertIsNone(state.model)
        self.assertFalse(state.verify(self.name))

    @patch("local_model_dock.probe_records.process_owns_loopback_port", return_value=True)
    @patch("local_model_dock.probe_records._request")
    def test_pass_persists_and_changed_artifact_invalidates(self, request, owns):
        request.return_value = self.response
        self.assertTrue(verify_model(self.backend, self.name))
        self.assertEqual(self.restarts, [True])
        self.assertEqual(request.call_args.args[1]["model"], self.name)
        self.assertTrue(check_record(self.backend, self.name))
        (self.root / "models" / self.name / "model.safetensors").write_bytes(b"changed")
        self.assertFalse(check_record(self.backend, self.name))

    @patch("local_model_dock.probe_records.process_owns_loopback_port", return_value=False)
    @patch("local_model_dock.probe_records._request")
    def test_unowned_listener_never_probes_or_records(self, request, owns):
        self.assertFalse(verify_model(self.backend, self.name))
        request.assert_not_called()
        self.assertFalse((self.root / "records").exists())

    @patch("local_model_dock.probe_records.process_owns_loopback_port", return_value=True)
    @patch("local_model_dock.probe_records._request")
    def test_failed_probe_does_not_create_record(self, request, owns):
        request.return_value = {"choices": []}
        self.assertFalse(verify_model(self.backend, self.name))
        self.assertFalse((self.root / "records").exists())

    @patch("local_model_dock.probe_records.process_owns_loopback_port", return_value=True)
    @patch("local_model_dock.probe_records._request")
    def test_changed_binary_invalidates_record(self, request, owns):
        request.return_value = self.response
        self.assertTrue(verify_model(self.backend, self.name))
        (self.root / "omlx").write_text("#!/bin/sh\nprintf '1.2.3\\n'\n# changed\n")
        self.assertFalse(check_record(self.backend, self.name))

    @patch("local_model_dock.probe_records.process_owns_loopback_port", return_value=True)
    @patch("local_model_dock.probe_records._request")
    def test_changed_backend_version_invalidates_record(self, request, owns):
        version_file = self.root / "version"
        version_file.write_text("1.2.3")
        (self.root / "omlx").write_text(f"#!/bin/sh\ncat '{version_file}'\n")
        request.return_value = self.response
        self.assertTrue(verify_model(self.backend, self.name))
        version_file.write_text("1.2.4")
        self.assertFalse(check_record(self.backend, self.name))

    @patch("local_model_dock.probe_records.process_owns_loopback_port")
    @patch("local_model_dock.probe_records._request")
    def test_listener_lost_during_probe_does_not_record(self, request, owns):
        owns.side_effect = [True, True, True, True, True, False]
        request.return_value = self.response
        self.assertFalse(verify_model(self.backend, self.name))
        self.assertFalse((self.root / "records").exists())

    @patch("local_model_dock.probe_records.process_owns_loopback_port", return_value=True)
    @patch("local_model_dock.probe_records._request")
    def test_failed_reprobe_invalidates_previous_record(self, request, owns):
        request.return_value = self.response
        self.assertTrue(verify_model(self.backend, self.name))
        request.return_value = {"choices": []}
        self.assertFalse(verify_model(self.backend, self.name))
        self.assertFalse(check_record(self.backend, self.name))

    @patch("local_model_dock.probe_records.process_owns_loopback_port", return_value=True)
    def test_probe_uses_loopback_http_and_model_directory_name(self, owns):
        seen = []
        response = self.response

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                seen.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                body = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self.addCleanup(server.server_close)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        self.backend.port = server.server_port
        self.assertTrue(verify_model(self.backend, self.name))
        self.assertEqual(seen[0][0], "/v1/chat/completions")
        self.assertEqual(seen[0][1]["model"], self.name)
