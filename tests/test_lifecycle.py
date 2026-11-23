import http.client
import concurrent.futures
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest


SOURCE = str(Path(__file__).resolve().parents[1] / "src")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        executable = self.root / "ollama"
        executable.write_text(
            "#!" + sys.executable + "\n"
            "import os,socket\n"
            "host,port=os.environ['OLLAMA_HOST'].split(':')\n"
            "s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind((host,int(port))); s.listen()\n"
            "while True:\n"
            " c,a=s.accept(); c.close()\n"
        )
        executable.chmod(0o700)
        (self.root / "omlx").write_text(
            "#!" + sys.executable + "\n"
            "import socket,sys\n"
            "s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(('127.0.0.1',int(sys.argv[sys.argv.index('--port')+1]))); s.listen()\n"
            "while True:\n"
            " c,a=s.accept(); c.close()\n"
        )
        (self.root / "omlx").chmod(0o700)
        self.port = free_port()
        self.env = {**os.environ, "PYTHONPATH": SOURCE, "PATH": str(self.root) + os.pathsep + os.environ["PATH"],
                    "API_KEY": "secret", "DOCK_BACKEND_PORT": str(free_port()),
                    "DOCK_GATEWAY_PORT": str(self.port)}
        self.addCleanup(self.stop)

    def command(self, command, env=None):
        return subprocess.run([sys.executable, "-m", "local_model_dock", command],
                              cwd=self.root, env=env or self.env, text=True,
                              capture_output=True, timeout=20)

    def stop(self):
        self.command("stop")

    def test_default_gateway_port(self):
        with socket.socket() as probe:
            try:
                probe.bind(("0.0.0.0", 8399))
            except OSError:
                self.skipTest("default gateway port is already in use")
        env = self.env.copy()
        del env["DOCK_GATEWAY_PORT"]
        start = self.command("start", env)
        self.assertEqual(start.returncode, 0, start.stderr)
        conn = http.client.HTTPConnection("127.0.0.1", 8399, timeout=3)
        self.addCleanup(conn.close)
        conn.request("GET", "/v1/models")
        self.assertEqual(conn.getresponse().status, 401)

    def test_start_duplicate_stop_and_restart_across_processes(self):
        start = self.command("start")
        self.assertEqual(start.returncode, 0, start.stderr)
        self.assertIn("no verified model", start.stdout.lower())
        self.assertNotEqual(self.command("start").returncode, 0)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        conn.request("GET", "/v1/models")
        self.assertEqual(conn.getresponse().status, 401)
        conn.close()
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        conn.request("POST", "/v1/chat/completions", '{"model":"local-model"}',
                     {"Authorization": "Bearer secret", "Content-Type": "application/json"})
        response = conn.getresponse()
        self.assertEqual(response.status, 503)
        self.assertIn(b'No verified model', response.read())
        conn.close()
        restarted = self.command("restart")
        self.assertEqual(restarted.returncode, 0, restarted.stderr)
        stopped = self.command("stop")
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", self.port), timeout=1)

    def test_missing_key_never_starts(self):
        result = self.command("start", {**self.env, "API_KEY": ""})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("API", result.stderr)

    def test_unsafe_directory_is_rejected(self):
        path = self.root / ".local-model-dock"
        path.mkdir(mode=0o755)
        result = self.command("start")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("private", result.stderr)

    def test_existing_non_socket_is_not_removed(self):
        path = self.root / ".local-model-dock"
        path.mkdir(mode=0o700)
        (path / "control.sock").write_text("keep")
        result = self.command("start")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((path / "control.sock").read_text(), "keep")

    def test_stale_socket_is_replaced(self):
        path = self.root / ".local-model-dock"
        path.mkdir(mode=0o700)
        with socket.socket(socket.AF_UNIX) as sock:
            sock.bind(str(path / "control.sock"))
        result = self.command("start")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_live_socket_without_lock_is_not_replaced(self):
        path = self.root / ".local-model-dock"
        path.mkdir(mode=0o700)
        with socket.socket(socket.AF_UNIX) as sock:
            sock.bind(str(path / "control.sock"))
            sock.listen()
            result = self.command("start")
            self.assertNotEqual(result.returncode, 0)
            with socket.socket(socket.AF_UNIX) as client:
                client.connect(str(path / "control.sock"))

    def test_concurrent_starts_allow_only_one_supervisor(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.command("start"), range(2)))
        self.assertEqual(sorted(result.returncode for result in results), [0, 1])
        self.assertEqual(self.command("stop").returncode, 0)

    def test_occupied_backend_port_fails_without_touching_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            env = {**self.env, "DOCK_BACKEND_PORT": str(listener.getsockname()[1])}
            result = self.command("start", env)
            self.assertNotEqual(result.returncode, 0)
            with socket.create_connection(listener.getsockname(), timeout=1):
                pass
        self.assertEqual(self.command("start").returncode, 0)

    def test_restart_after_backend_closes_connection_first(self):
        started = self.command("start")
        self.assertEqual(started.returncode, 0, started.stderr)
        with socket.create_connection(("127.0.0.1", int(self.env["DOCK_BACKEND_PORT"])), timeout=2) as client:
            self.assertEqual(client.recv(1), b"")
        restarted = self.command("restart")
        self.assertEqual(restarted.returncode, 0, restarted.stderr)

    def test_wildcard_listener_is_not_mistaken_for_available_backend(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("0.0.0.0", 0))
            listener.listen()
            env = {**self.env, "DOCK_BACKEND_PORT": str(listener.getsockname()[1])}
            result = self.command("start", env)
            self.assertNotEqual(result.returncode, 0)
            with socket.create_connection(("127.0.0.1", listener.getsockname()[1]), timeout=1):
                pass
