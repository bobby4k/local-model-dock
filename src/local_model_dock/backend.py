"""Foreground child-process backends; the caller retains ownership of the child."""

import os
import shutil
import subprocess
from pathlib import Path


class BackendError(RuntimeError):
    pass


class Backend:
    def __init__(self, kind: str, root: Path, port: int = 11434, cache_gb: int = 1):
        if kind not in ("ollama", "omlx"):
            raise ValueError("backend must be ollama or omlx")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if type(cache_gb) is not int or cache_gb <= 0:
            raise ValueError("cache_gb must be a positive integer")
        self.kind, self.root, self.port, self.cache_gb = kind, Path(root).resolve(), port, cache_gb
        self.process = None

    def start(self):
        """Spawn a child, not a readiness check; retain this object to stop it."""
        if self.process is not None:
            if self.process.poll() is None:
                raise BackendError("backend is already running")
            self.process = None
        executable = shutil.which(self.kind)
        if not executable:
            raise BackendError(f"{self.kind} executable not found on PATH")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in ("models", "cache"):
            path = self.root / name
            if not path.resolve().is_relative_to(self.root):
                raise BackendError(f"{name} path escapes project root")
            path.mkdir(mode=0o700, exist_ok=True)
            if path.stat().st_mode & 0o077:
                raise BackendError(f"{name} directory must be private (mode 700)")
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("OMLX_", "OLLAMA_"))}
        env.update(HOME=str(self.root), HF_HOME=str(self.root / "huggingface"),
                   XDG_CACHE_HOME=str(self.root / "cache"))
        if self.kind == "ollama":
            env.update(OLLAMA_HOST=f"127.0.0.1:{self.port}",
                       OLLAMA_MODELS=str(self.root / "models"), OLLAMA_NO_CLOUD="1",
                       OLLAMA_NOPRUNE="1")
            command = [executable, "serve"]
        else:
            command = [executable, "serve", "--host", "127.0.0.1",
                       "--port", str(self.port), "--base-path", str(self.root),
                       "--model-dir", str(self.root / "models"),
                       "--paged-ssd-cache-dir", str(self.root / "cache"),
                       "--paged-ssd-cache-max-size", f"{self.cache_gb}GB", "--no-hf-cache"]
        try:
            self.process = subprocess.Popen(command, env=env, cwd=self.root,
                                            stdin=subprocess.DEVNULL)
        except OSError as exc:
            raise BackendError(f"could not start {self.kind}: {exc}") from exc
        if self.process.poll() is not None:
            code = self.process.returncode
            self.process = None
            raise BackendError(f"{self.kind} exited during startup ({code})")

    def stop(self):
        """Stop only the direct child started by this instance."""
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BackendError(f"could not stop {self.kind}: {exc}") from exc
        self.process = None

    def restart(self):
        self.stop()
        self.start()
