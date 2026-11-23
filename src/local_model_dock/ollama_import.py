"""Fail-closed inspection of downloaded GGUF sources for Ollama."""

import hashlib
import json
import re
import stat
from pathlib import Path

from .artifact_verification import MANIFEST


def source_gguf(root, name):
    """Return source identity only for one private, SHA256-verified GGUF."""
    root = Path(root)
    if (not isinstance(name, str) or "/" in name or "\\" in name
            or name in ("", ".", "..") or "\0" in name):
        raise ValueError("invalid source name")
    models = root / "models"
    model = models / name
    for directory in (root, models, model):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("source directory must be private")
    manifest_path = model / MANIFEST
    info = manifest_path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 10_000_000:
        raise ValueError("invalid source manifest")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    repo, revision = manifest.get("repo"), manifest.get("revision")
    if (manifest.get("version") != 1 or not isinstance(repo, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo)
            or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or name != repo.replace("/", "--") + "--" + revision
            or not isinstance(manifest.get("files"), list)):
        raise ValueError("invalid source identity")
    entries = [entry for entry in manifest["files"] if isinstance(entry, dict)
               and isinstance(entry.get("path"), str) and entry["path"].lower().endswith(".gguf")]
    if len(entries) != 1:
        raise ValueError("exactly one GGUF is required")
    entry = entries[0]
    filename, digest, size = entry["path"], entry.get("digest"), entry.get("size")
    if ("/" in filename or "\\" in filename or filename in (".", "..")
            or "\0" in filename or entry.get("digest_type") != "sha256"
            or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or type(size) is not int or size <= 0):
        raise ValueError("invalid GGUF manifest entry")
    path = model / filename
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != size:
        raise ValueError("invalid GGUF file")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError("GGUF digest mismatch")
    return {"repo": repo, "revision": revision, "filename": filename,
            "sha256": digest, "manifest_sha256": hashlib.sha256(raw).hexdigest()}
