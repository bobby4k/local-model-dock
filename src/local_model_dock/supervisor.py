"""Private, lock-protected supervisor; only it owns and stops its backend child."""

import fcntl
import errno
import json
import os
import select
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .backend import Backend, BackendError
from .config import load_config
from .gateway import create_gateway
from .hardware import detect_backend
from .probe_records import OLLAMA_MODELS, check_record, checked_identity, verify_model


ROOT = Path(".local-model-dock")
SOCKET = "control.sock"


class ModelState:
    def __init__(self, backend, persist=None, hash_cache_ttl=86400):
        self.backend = backend
        self.persist = persist
        self.lock = threading.Lock()
        self.validation_lock = threading.Lock()
        self.hash_cache = {}
        self.hash_cache_ttl = hash_cache_ttl
        self.model = None
        self.active = 0
        self.busy = False

    @contextmanager
    def request(self):
        with self.lock:
            model = None if self.busy else self.model
            acquired = model is not None
            if acquired:
                self.active += 1
        try:
            with self.validation_lock:
                valid = model is None or check_record(self.backend, model, self.hash_cache, self.hash_cache_ttl)
            if not valid:
                with self.lock:
                    if self.model == model:
                        self.model = None
                model = None
            yield model
        finally:
            if acquired:
                with self.lock:
                    self.active -= 1

    def enable(self, model):
        return self.operation("enable", model)

    def verify(self, model):
        return self.operation("verify", model)

    def reserve(self, action):
        with self.lock:
            if self.busy or (action in ("verify", "enable", "stop") and self.active):
                return False
            self.busy = True
            if action in ("verify", "enable", "stop"):
                self.hash_cache.clear()
            if action in ("verify", "stop"):
                self.model = None
            return True

    def operation(self, action, model, reserved=False):
        if not reserved and not self.reserve(action):
            return False
        try:
            operation = {"verify": verify_model, "check": checked_identity, "enable": check_record}[action]
            valid = operation(self.backend, model)
            if action == "enable":
                if self.persist:
                    self.persist(model if valid else None)
                with self.lock:
                    self.model = model if valid else None
            return valid
        finally:
            with self.lock:
                self.busy = False


def load_enabled(root):
    path = root / "enabled-model"
    try:
        info = path.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or info.st_size > 128):
            return None
        value = path.read_text()
        return value if value in OLLAMA_MODELS else None
    except (OSError, UnicodeError):
        return None


def save_enabled(root, model):
    path = root / "enabled-model"
    if model is None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    if model not in OLLAMA_MODELS:
        raise ValueError("unsupported persistent model")
    fd, temporary = tempfile.mkstemp(prefix=".enabled-", dir=root)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(model)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def private_root(create=False):
    if create:
        ROOT.mkdir(mode=0o700, exist_ok=True)
    info = ROOT.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o077):
        raise ValueError("runtime directory must be private and owned by the current user")
    return ROOT.resolve()


def start():
    config = load_config()
    if not config.api_key.strip() or "\n" in config.api_key or "\r" in config.api_key:
        raise ValueError("API_KEY must be configured")
    private_root(create=True)
    process = subprocess.Popen([sys.executable, "-m", "local_model_dock.supervisor"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    ready, _, _ = select.select([process.stdout], [], [], 120)
    if not ready:
        process.terminate()
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            raise RuntimeError("supervisor startup timed out; cleanup is still pending")
        raise RuntimeError("supervisor startup timed out")
    line = process.stdout.readline().decode("utf-8", "replace").strip()
    process.stdout.close()
    if line not in ("READY", "READY ENABLED"):
        try:
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{line or 'startup failed'}; cleanup is still pending") from None
        if line == "Startup failed: supervisor already running":
            path = private_root() / SOCKET
            info = path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("control path is not an owned socket")
            with socket.socket(socket.AF_UNIX) as client:
                client.settimeout(2)
                client.connect(str(path))
                client.sendall(b"status\n")
                response = client.recv(128).decode("ascii").strip()
            if response.isdecimal() and 1 <= int(response) <= 65535:
                print(f"Already running. API URL: http://127.0.0.1:{int(response)}/v1")
                return
        raise RuntimeError(line or "supervisor exited during startup")
    print("Started authenticated gateway; " + ("verified model restored." if line == "READY ENABLED"
                                               else "no verified model is available."))
    print(f"API URL: http://127.0.0.1:{_port('DOCK_GATEWAY_PORT', '8399')}/v1")


def stop():
    root = private_root()
    path = root / SOCKET
    info = path.lstat()
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("control path is not an owned socket")
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(15)
        client.connect(str(path))
        client.sendall(b"stop\n")
        response = client.recv(4096).decode("utf-8", "replace").strip()
    if response != "STOPPED":
        raise RuntimeError(response or "supervisor did not confirm shutdown")
    print("Stopped.")


def _port(name, default):
    value = int(os.environ.get(name, default))
    if not 1 <= value <= 65535:
        raise ValueError(f"{name} must be between 1 and 65535")
    return value


def run():
    # Keep the startup channel separate from backend output.
    status = os.fdopen(os.dup(1), "w", buffering=1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    backend = gateway = control = lock = None
    workers = []
    path = None
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        root = private_root(create=True)
        lock_fd = os.open(root / "supervisor.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        lock = os.fdopen(lock_fd, "r+")
        info = os.fstat(lock_fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("supervisor lock must be a private owned file")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("supervisor already running") from None
        path = root / SOCKET
        if path.exists() or path.is_symlink():
            info = path.lstat()
            if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
                raise ValueError("control path is not an owned socket")
            with socket.socket(socket.AF_UNIX) as probe:
                probe.settimeout(1)
                try:
                    probe.connect(str(path))
                except OSError as exc:
                    if exc.errno != errno.ECONNREFUSED:
                        raise RuntimeError("cannot verify stale control socket") from exc
                else:
                    raise RuntimeError("control socket is already in use")
            path.unlink()
        config = load_config()
        backend_port = _port("DOCK_BACKEND_PORT", "11434")
        gateway_port = _port("DOCK_GATEWAY_PORT", "8399")
        # Fail rather than attaching to a pre-existing, potentially unrelated backend.
        for host in ("127.0.0.1", "0.0.0.0"):
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((host, backend_port))
        backend = Backend(detect_backend(), root, port=backend_port)
        state = ModelState(backend, persist=lambda model: save_enabled(
            root, model if model in OLLAMA_MODELS else None),
            hash_cache_ttl=config.ollama_hash_cache_ttl_seconds)
        gateway = create_gateway(("0.0.0.0", gateway_port),
                                 f"http://127.0.0.1:{backend_port}", config.api_key,
                                 "local-model", lambda: None, model_session=state.request)
        backend.start()
        deadline = time.monotonic() + 10
        while True:
            if stopping:
                raise RuntimeError("startup cancelled")
            if backend.process.poll() is not None:
                raise RuntimeError("backend exited before becoming ready")
            try:
                with socket.create_connection(("127.0.0.1", backend_port), timeout=0.2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("backend readiness timed out")
                time.sleep(0.05)
        selected = load_enabled(root)
        if selected:
            state.enable(selected)
        control = socket.socket(socket.AF_UNIX)
        control.bind(str(path))
        os.chmod(path, 0o600)
        control.listen(4)
        control.settimeout(0.2)
        thread = threading.Thread(target=gateway.serve_forever, daemon=True)
        thread.start()
        status.write("READY ENABLED\n" if state.model else "READY\n")
        status.close()
        while not stopping:
            with state.lock:
                busy = state.busy
            if not busy and (backend.process is None or backend.process.poll() is not None):
                break
            try:
                client, _ = control.accept()
            except socket.timeout:
                continue
            with client:
                client.settimeout(2)
                stop_reserved = False
                try:
                    request = b""
                    while not request.endswith(b"\n") and len(request) <= 4096:
                        chunk = client.recv(4097 - len(request))
                        if not chunk:
                            break
                        request += chunk
                    if request == b"verify\n":
                        client.sendall(b"Verification unavailable: backend and model identity cannot be proven\n")
                        continue
                    if request == b"status\n":
                        client.sendall(f"{gateway_port}\n".encode())
                        continue
                    if request.startswith(b"{") and request.endswith(b"\n") and len(request) <= 4096:
                        try:
                            command = json.loads(request)
                            if (not isinstance(command, dict) or set(command) != {"action", "model"}
                                    or command["action"] not in ("verify", "check", "enable")
                                    or not isinstance(command["model"], str)):
                                raise ValueError("invalid control request")
                            action, model = command["action"], command["model"]
                            if not state.reserve(action):
                                client.sendall(b'{"verified":false,"busy":true}\n')
                                continue
                            connection = client.dup()
                            def execute_control(connection, action, model):
                                with connection:
                                    try:
                                        result = state.operation(action, model, reserved=True)
                                        if action == "check":
                                            response = ({"verified": True, "identity": result}
                                                        if isinstance(result, dict) else {"verified": False})
                                        else:
                                            response = {"verified": result is True}
                                        connection.sendall(json.dumps(response).encode() + b"\n")
                                    except Exception:
                                        try:
                                            connection.sendall(b'{"verified":false}\n')
                                        except OSError:
                                            pass
                            workers = [item for item in workers if item.is_alive()]
                            try:
                                worker = threading.Thread(target=execute_control,
                                                          args=(connection, action, model), daemon=True)
                                worker.start()
                            except Exception:
                                connection.close()
                                with state.lock:
                                    state.busy = False
                                raise
                            workers.append(worker)
                        except (ValueError, TypeError, UnicodeError):
                            client.sendall(b'{"verified":false}\n')
                        continue
                    if request != b"stop\n":
                        client.sendall(b"Invalid request\n")
                        continue
                    if not state.reserve("stop"):
                        client.sendall(b"Busy: requests or verification in progress\n")
                        continue
                    stop_reserved = True
                    backend.stop()
                    gateway.shutdown()
                    gateway.server_close()
                    gateway = None
                    path.unlink()
                    path = None
                    lock.close()
                    lock = None
                    client.sendall(b"STOPPED\n")
                    return
                except (OSError, BackendError) as exc:
                    if stop_reserved:
                        with state.lock:
                            state.busy = False
                    try:
                        client.sendall(f"Shutdown failed: {exc}\n".encode())
                    except OSError:
                        pass
        if backend.process is None or backend.process.poll() is not None:
            raise RuntimeError("backend exited unexpectedly")
    except (OSError, ValueError, RuntimeError, BackendError) as exc:
        if not status.closed:
            status.write(f"Startup failed: {exc}\n")
    finally:
        for worker in workers:
            worker.join()
        if backend is not None:
            try:
                backend.stop()
            except BackendError:
                # Retain ownership and the lock until the child actually exits.
                if backend.process is not None:
                    backend.process.wait()
        if gateway is not None:
            gateway.server_close()
        if control is not None:
            control.close()
        if path is not None and control is not None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        if lock is not None:
            lock.close()
        if not status.closed:
            status.close()


if __name__ == "__main__":
    run()
