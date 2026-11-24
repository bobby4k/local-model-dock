"""Fail-closed tool-call probing for the supervisor's own oMLX child."""

import hashlib
import http.client
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

from .artifact_verification import MANIFEST, verify_artifact
from .validation import validate_tool_call

from .process_identity import process_owns_loopback_port

OLLAMA_MODEL = "deepseek-v4-pro-qwen3.5-9b-uncensored:q4_k_m"
OLLAMA_MODELS = (OLLAMA_MODEL, "local-dock-initial")


def _ollama_parts(name):
    if not isinstance(name, str) or not re.fullmatch(
            r"[a-z0-9][a-z0-9._-]*(?::[a-z0-9][a-z0-9._-]*)?", name):
        raise ValueError("invalid Ollama model name")
    parts = name.split(":")
    return parts[0], parts[1] if len(parts) == 2 else "latest"


def _ollama_layer(root, name, cache=None):
    """Hash the single model layer of a local, unqualified Ollama model."""
    model, tag = _ollama_parts(name)
    models = Path(root) / "models"
    manifest_dir = models / "manifests-v2" / "ollama.com" / "library" / model
    for path in (models, models / "manifests-v2", models / "manifests-v2" / "ollama.com",
                 models / "manifests-v2" / "ollama.com" / "library", manifest_dir,
                 models / "blobs"):
        if path.is_symlink() or not path.is_dir():
            raise ValueError("invalid Ollama model directory")
    manifest_path = manifest_dir / tag
    manifest_digest = None
    if manifest_path.is_symlink():
        target = manifest_path.resolve(strict=True)
        if target.parent != (models / "blobs").resolve() or not re.fullmatch(r"sha256-[0-9a-f]{64}", target.name):
            raise ValueError("invalid Ollama manifest link")
        manifest_path = target
        manifest_digest = target.name[7:]
    if not manifest_path.is_file() or manifest_path.stat().st_size > 1_000_000:
        raise ValueError("invalid Ollama manifest")
    raw = manifest_path.read_bytes()
    if manifest_digest is not None and hashlib.sha256(raw).hexdigest() != manifest_digest:
        raise ValueError("Ollama manifest digest mismatch")
    manifest = json.loads(raw)
    layers = manifest["layers"]
    if not isinstance(layers, list):
        raise ValueError("invalid Ollama layers")
    model_layers = [layer for layer in layers if isinstance(layer, dict)
                    and layer.get("mediaType") == "application/vnd.ollama.image.model"]
    if len(model_layers) != 1:
        raise ValueError("expected exactly one Ollama model layer")
    descriptors = layers + ([manifest["config"]] if "config" in manifest else [])
    for layer in descriptors:
        if not isinstance(layer, dict):
            raise ValueError("invalid Ollama descriptor")
        digest, size = layer.get("digest"), layer.get("size")
        if (not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
                or type(size) is not int or size <= 0):
            raise ValueError("invalid Ollama layer identity")
        blob = models / "blobs" / digest.replace(":", "-")
        if blob.is_symlink() or not blob.is_file() or blob.stat().st_size != size:
            raise ValueError("invalid Ollama layer blob")
        with blob.open("rb") as stream:
            before = os.fstat(stream.fileno())
            identity = (before.st_dev, before.st_ino, before.st_size,
                        before.st_mtime_ns, before.st_ctime_ns)
            entry = cache.get(str(blob)) if cache is not None else None
            now = time.monotonic()
            # shortcut: unchanged metadata reuses a digest for 30 seconds; rehash for stronger detection.
            if entry and entry[0] == identity and now < entry[1]:
                actual = entry[2]
            else:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
                after = os.fstat(stream.fileno())
                if identity != (after.st_dev, after.st_ino, after.st_size,
                                after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError("Ollama layer changed during verification")
                if cache is not None and actual == digest[7:]:
                    if len(cache) >= 64:
                        cache.clear()
                    cache[str(blob)] = (identity, now + 30, actual)
        if actual != digest[7:]:
            raise ValueError("Ollama layer digest mismatch")
    return model_layers[0]["digest"][7:], model_layers[0]["size"]


def _model_limits_verified(model, manifest):
    # shortcut: no trusted same-revision parameter evidence is persisted; enable after adding it.
    return False


def _request(port, payload):
    return _post(port, "/v1/chat/completions", payload)


def _ollama_show(port, name):
    return _post(port, "/api/show", {"model": name, "verbose": True}, limit=32_000_000)


def _post(port, path, payload, limit=1_000_000):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        conn.request("POST", path, json.dumps(payload),
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read(limit + 1)
        if response.status != 200 or len(data) > limit:
            raise ValueError("backend probe failed")
        return json.loads(data)
    finally:
        conn.close()


def _identity(backend, name, cache=None):
    if backend.kind == "ollama":
        return _ollama_identity(backend, name, cache)
    if (not isinstance(name, str) or not name or name in (".", "..")
            or "/" in name or "\\" in name or "\0" in name or backend.kind != "omlx"):
        raise ValueError("unsupported model or backend")
    process = backend.process
    if process is None or process.poll() is not None or not process_owns_loopback_port(process, backend.port):
        raise ValueError("backend listener identity unavailable")
    args = process.args
    if (not isinstance(args, (tuple, list)) or len(args) < 2 or args[1] != "serve"
            or args.count("--model-dir") != 1 or "--no-hf-cache" not in args
            or args[args.index("--model-dir") + 1] != str(backend.root / "models")):
        raise ValueError("backend model directory not established")
    executable = Path(args[0])
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("backend executable unavailable")
    executable = executable.resolve()
    with executable.open("rb") as stream:
        binary_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    version = subprocess.run([str(executable), "--version"], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, timeout=10, check=True).stdout.decode().strip()
    if not re.fullmatch(r"[0-9][0-9A-Za-z.+_-]*", version):
        raise ValueError("invalid backend version")
    models = backend.root / "models"
    if models.is_symlink() or not models.is_dir():
        raise ValueError("invalid models directory")
    model = models / name
    manifest = verify_artifact(model)
    if name != manifest["repo"].replace("/", "--") + "--" + manifest["revision"]:
        raise ValueError("model identity mismatch")
    if not _model_limits_verified(model, manifest):
        raise ValueError("model version and parameter limits cannot be proven")
    manifest_hash = hashlib.sha256((model / MANIFEST).read_bytes()).hexdigest()
    if process.poll() is not None or not process_owns_loopback_port(process, backend.port):
        raise ValueError("backend listener changed")
    return {"model": name, "repo": manifest["repo"], "revision": manifest["revision"],
            "manifest_sha256": manifest_hash, "backend": backend.kind,
            "backend_version": version, "binary": str(executable), "binary_sha256": binary_hash}


def _ollama_identity(backend, name, cache=None):
    # Fixed HF tree revision c6453f674da4d30f5737092d9371114cae3ca063, Q4_K_M LFS entry.
    repo = "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF"
    revision = "c6453f674da4d30f5737092d9371114cae3ca063"
    digest = "de3fea7a12b793a573df636eb1c16da3f7960af0377f79e04bfc04a8e65dfb4c"
    if name not in OLLAMA_MODELS:
        raise ValueError("Ollama model is not the authorized fallback")
    process = backend.process
    if process is None or process.poll() is not None or not process_owns_loopback_port(process, backend.port):
        raise ValueError("backend listener identity unavailable")
    args = process.args
    if not isinstance(args, (tuple, list)) or len(args) != 2 or args[1] != "serve":
        raise ValueError("invalid Ollama child command")
    executable = Path(args[0])
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("backend executable unavailable")
    executable = executable.resolve()
    with executable.open("rb") as stream:
        binary_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    env = {key: value for key, value in os.environ.items() if not key.startswith("OLLAMA_")}
    env["OLLAMA_HOST"] = f"127.0.0.1:{backend.port}"
    output = subprocess.run([str(executable), "--version"], stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, timeout=10, check=True, env=env).stdout.decode().strip()
    match = re.fullmatch(r"ollama version is ([0-9][0-9A-Za-z.+_-]*)", output)
    if not match or _ollama_layer(backend.root, name, cache) != (digest, 5780093088):
        raise ValueError("Ollama version or model digest mismatch")
    model, tag = _ollama_parts(name)
    manifest = backend.root / "models" / "manifests-v2" / "ollama.com" / "library" / model / tag
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    show = _ollama_show(backend.port, name)
    info, details = show["model_info"], show["details"]
    count = info["general.parameter_count"]
    if (type(count) is not int or count < 9_000_000_000
            or info.get("general.architecture") != "qwen35" or details.get("family") != "qwen35"):
        raise ValueError("Ollama model policy evidence mismatch")
    if process.poll() is not None or not process_owns_loopback_port(process, backend.port):
        raise ValueError("backend listener changed")
    return {"model": name, "repo": repo, "revision": revision, "model_sha256": digest,
            "manifest_sha256": manifest_hash, "parameter_count": count, "backend": "ollama",
            "backend_version": match.group(1), "binary": str(executable), "binary_sha256": binary_hash}


def _record_path(backend, name):
    return backend.root / "records" / (hashlib.sha256(name.encode()).hexdigest() + ".json")


def verify_model(backend, name):
    try:
        if not isinstance(name, str):
            return False
        records = backend.root / "records"
        if records.is_symlink():
            return False
        previous = _record_path(backend, name)
        if previous.is_symlink():
            return False
        if previous.exists():
            previous.unlink()
        before = _identity(backend, name)
        # A fresh child prevents probing an older model already cached in memory.
        backend.restart()
        deadline = time.monotonic() + 15
        while not process_owns_loopback_port(backend.process, backend.port):
            if backend.process is None or backend.process.poll() is not None or time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        if _identity(backend, name) != before:
            return False
        def request(payload):
            payload["model"] = name
            result = _request(backend.port, payload)
            returned_name = result.get("model") if isinstance(result, dict) else None
            if returned_name != name and not (backend.kind == "ollama" and returned_name == name + ":latest"):
                raise ValueError("backend returned a different model")
            return result
        result = validate_tool_call(request, before["revision"], before["backend"],
                                    before["backend_version"])
        if not result.passed or _identity(backend, name) != before:
            return False
        records.mkdir(mode=0o700, exist_ok=True)
        if records.is_symlink() or not records.is_dir() or records.stat().st_mode & 0o077:
            return False
        fd, temporary = tempfile.mkstemp(dir=records, prefix=".record-")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(before, stream, sort_keys=True)
            os.replace(temporary, _record_path(backend, name))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return True
    except (OSError, RuntimeError, ValueError, TypeError, IndexError, KeyError, UnicodeError,
            subprocess.SubprocessError, http.client.HTTPException):
        return False


def checked_identity(backend, name, cache=None):
    """Return the validated public identity, or None if any evidence is stale."""
    try:
        identity = _identity(backend, name, cache)
        records = backend.root / "records"
        if records.is_symlink() or not records.is_dir() or records.stat().st_mode & 0o077:
            return None
        path = _record_path(backend, name)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 100_000:
            return None
        if json.loads(path.read_bytes()) != identity:
            return None
        return {key: identity[key] for key in ("repo", "revision", "backend", "backend_version")}
    except (OSError, ValueError, TypeError, IndexError, KeyError, UnicodeError,
            subprocess.SubprocessError, http.client.HTTPException):
        return None


def check_record(backend, name, cache=None):
    return checked_identity(backend, name, cache) is not None
