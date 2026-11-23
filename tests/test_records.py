import os
import json
import http.client
import hashlib
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_lifecycle import SOURCE, free_port


class VerificationControlTests(unittest.TestCase):
    def test_check_returns_bound_identity_and_rejects_changed_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dock = root / ".local-model-dock"
            model_name = "owner--model--" + "a" * 40
            model = dock / "models" / model_name
            model.mkdir(parents=True, mode=0o700)
            dock.chmod(0o700)
            (dock / "models").chmod(0o700)
            entries = []
            for name, data in (("config.json", b"{}"), ("model.safetensors", b"weights")):
                (model / name).write_bytes(data)
                entries.append({"path": name, "size": len(data), "digest_type": "sha256",
                                "digest": hashlib.sha256(data).hexdigest()})
            (model / ".local-model-dock-manifest.json").write_text(json.dumps({
                "version": 1, "repo": "owner/model", "revision": "a" * 40, "files": entries}))
            executable = root / "omlx"
            executable.write_text(
                "#!" + sys.executable + "\n"
                "import json,sys\n"
                "if '--version' in sys.argv: print('1.2.3'); sys.exit()\n"
                "from http.server import BaseHTTPRequestHandler,HTTPServer\n"
                "class H(BaseHTTPRequestHandler):\n"
                " def log_message(self,*a): pass\n"
                " def do_POST(self):\n"
                "  p=json.loads(self.rfile.read(int(self.headers['Content-Length'])))\n"
                "  b=json.dumps({'model':p['model'],'choices':[{'message':{'tool_calls':[{'type':'function','function':{'name':'lookup','arguments':'{\\\"query\\\":\\\"dock\\\"}'}}]}}]}).encode()\n"
                "  self.send_response(200); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)\n"
                "HTTPServer(('127.0.0.1',int(sys.argv[sys.argv.index('--port')+1])),H).serve_forever()\n"
            )
            executable.chmod(0o700)
            env = {**os.environ, "PYTHONPATH": SOURCE, "PATH": str(root) + os.pathsep + os.environ["PATH"],
                   "API_KEY": "secret", "DOCK_BACKEND_PORT": str(free_port()),
                   "DOCK_GATEWAY_PORT": str(free_port())}
            code = ("import socket; import local_model_dock.supervisor as s; "
                     "import local_model_dock.probe_records as p; "
                     "s.detect_backend=lambda:'omlx'; "
                     "p._model_limits_verified=lambda model,manifest: True; "
                     "p.process_owns_loopback_port=lambda process,port: process is not None and process.poll() is None and socket.socket().connect_ex(('127.0.0.1',port)) == 0; s.run()")
            process = subprocess.Popen([sys.executable, "-c", code], cwd=root, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.addCleanup(lambda: process.kill() if process.poll() is None else None)
            self.assertEqual(process.stdout.readline(), b"READY\n")
            path = str(dock / "control.sock")
            def control(action):
                with socket.socket(socket.AF_UNIX) as client:
                    client.settimeout(30)
                    client.connect(path)
                    client.sendall(json.dumps({"action": action, "model": model_name}).encode() + b"\n")
                    return json.loads(client.recv(4096))
            try:
                self.assertTrue(control("verify")["verified"])
                response = control("check")
                self.assertEqual(response, {"verified": True, "identity": {
                    "repo": "owner/model", "revision": "a" * 40,
                    "backend": "omlx", "backend_version": "1.2.3"}})
                legacy = subprocess.run(
                    [sys.executable, "-c",
                     "from local_model_dock.cli import _control; import sys; _control('check',sys.argv[1])",
                     model_name], cwd=root, env=env, capture_output=True, timeout=30)
                self.assertEqual(legacy.returncode, 0, legacy.stderr)
                record = dock / "records" / (hashlib.sha256(model_name.encode()).hexdigest() + ".json")
                saved = record.read_bytes()
                tampered = json.loads(saved)
                tampered["revision"] = "b" * 40
                record.write_text(json.dumps(tampered))
                self.assertEqual(control("check"), {"verified": False})
                record.write_bytes(saved)
                (model / "model.safetensors").write_bytes(b"changed")
                self.assertEqual(control("check"), {"verified": False})
                legacy = subprocess.run(
                    [sys.executable, "-c",
                     "from local_model_dock.cli import _control; import sys; _control('check',sys.argv[1])",
                     model_name], cwd=root, env=env, capture_output=True, timeout=30)
                self.assertNotEqual(legacy.returncode, 0)
            finally:
                with socket.socket(socket.AF_UNIX) as client:
                    client.settimeout(5)
                    client.connect(path)
                    client.sendall(b"stop\n")
                    client.recv(4096)
                process.wait(timeout=5)
                process.stdout.close()
                process.stderr.close()

    def test_enable_is_async_and_gateway_uses_selected_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "fake"
            executable.write_text(
                "#!" + sys.executable + "\n"
                "import os,socket,sys\n"
                "port=int(sys.argv[sys.argv.index('--port')+1]) if '--port' in sys.argv else int(os.environ['OLLAMA_HOST'].split(':')[1])\n"
                "s=socket.socket(); s.bind(('127.0.0.1',port)); s.listen()\n"
                "while True:\n c,a=s.accept(); c.close()\n"
            )
            executable.chmod(0o700)
            (root / "omlx").symlink_to(executable)
            (root / "ollama").symlink_to(executable)
            port = free_port()
            env = {**os.environ, "PYTHONPATH": SOURCE, "PATH": str(root) + os.pathsep + os.environ["PATH"],
                   "API_KEY": "secret", "DOCK_BACKEND_PORT": str(free_port()),
                   "DOCK_GATEWAY_PORT": str(port)}
            code = ("import time; import local_model_dock.supervisor as s; "
                    "s.check_record=lambda *args: (time.sleep(1) or True); s.run()")
            process = subprocess.Popen([sys.executable, "-c", code], cwd=root, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.addCleanup(lambda: process.kill() if process.poll() is None else None)
            self.assertEqual(process.stdout.readline(), b"READY\n")
            path = str(root / ".local-model-dock" / "control.sock")
            with socket.socket(socket.AF_UNIX) as first:
                first.settimeout(5)
                first.connect(path)
                first.sendall(b'{"action":"enable","model":"candidate"}\n')
                with socket.socket(socket.AF_UNIX) as second:
                    second.settimeout(2)
                    second.connect(path)
                    second.sendall(b'{"action":"check","model":"candidate"}\n')
                    self.assertTrue(json.loads(second.recv(4096))["busy"])
                self.assertTrue(json.loads(first.recv(4096))["verified"])
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
            conn.request("GET", "/v1/models", headers={"Authorization": "Bearer secret"})
            self.assertEqual(json.loads(conn.getresponse().read())["data"][0]["id"], "local-model")
            conn.close()
            with socket.socket(socket.AF_UNIX) as client:
                client.settimeout(3)
                client.connect(path)
                client.sendall(b"stop\n")
                self.assertEqual(client.recv(4096), b"STOPPED\n")
            process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()

    def test_verify_fails_closed_when_backend_identity_cannot_be_proven(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = root / "omlx"
            backend.write_text(
                "#!" + sys.executable + "\n"
                "import os,socket,sys\n"
                "port=int(sys.argv[sys.argv.index('--port')+1]) if '--port' in sys.argv else int(os.environ['OLLAMA_HOST'].split(':')[1])\n"
                "s=socket.socket(); s.bind(('127.0.0.1',port)); s.listen()\n"
                "while True:\n"
                " c,a=s.accept(); c.close()\n"
            )
            backend.chmod(0o700)
            env = {**os.environ, "PYTHONPATH": SOURCE, "PATH": str(root) + os.pathsep + os.environ["PATH"],
                   "API_KEY": "secret", "DOCK_BACKEND_PORT": str(free_port()),
                   "DOCK_GATEWAY_PORT": str(free_port())}
            def command(name):
                return subprocess.run([sys.executable, "-m", "local_model_dock", name],
                                      cwd=root, env=env, capture_output=True, timeout=20)

            # Force the fake executable to be selected independent of the host OS.
            backend.rename(root / "ollama")
            (root / "omlx").symlink_to(root / "ollama")
            started = command("start")
            self.assertEqual(started.returncode, 0, started.stderr)
            try:
                with socket.socket(socket.AF_UNIX) as client:
                    client.settimeout(3)
                    client.connect(str(root / ".local-model-dock" / "control.sock"))
                    client.sendall(b"verify\n")
                    self.assertIn(b"identity", client.recv(4096).lower())
                with socket.socket(socket.AF_UNIX) as client:
                    client.settimeout(3)
                    client.connect(str(root / ".local-model-dock" / "control.sock"))
                    client.sendall(b'{"action":"verify","model":"missing"}\n')
                    self.assertFalse(json.loads(client.recv(4096))["verified"])
                self.assertFalse((root / ".local-model-dock" / "records").exists())
            finally:
                command("stop")
