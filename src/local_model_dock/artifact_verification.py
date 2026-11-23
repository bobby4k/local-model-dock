"""Verify downloaded model files against their download-time manifest."""

import hashlib
import json
import os
import re
import stat
from pathlib import Path


MANIFEST = ".local-model-dock-manifest.json"


def verify_artifact(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("invalid model directory")
    manifest_path = directory / MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("missing model manifest")
    if manifest_path.stat().st_size > 10_000_000:
        raise ValueError("model manifest too large")
    manifest = json.loads(manifest_path.read_bytes())
    if (not isinstance(manifest, dict) or type(manifest.get("version")) is not int
            or manifest["version"] != 1
            or not isinstance(manifest.get("repo"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", manifest["repo"])
            or not isinstance(manifest.get("revision"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"])
            or not isinstance(manifest.get("files"), list)):
        raise ValueError("invalid model manifest")
    expected = {}
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise ValueError("invalid manifest entry")
        name, size, kind, digest = (entry.get(k) for k in ("path", "size", "digest_type", "digest"))
        if (not isinstance(name, str) or not name or "\\" in name or "\0" in name
                or any(part in ("", ".", "..") for part in name.split("/"))
                or name == MANIFEST or name in expected or type(size) is not int or size < 0
                or kind not in ("sha256", "git-blob-sha1") or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}" if kind == "sha256" else r"[0-9a-f]{40}", digest)):
            raise ValueError("invalid manifest entry")
        expected[name] = (size, kind, digest)
    if "config.json" not in expected or not any(n.endswith(".safetensors") for n in expected):
        raise ValueError("model configuration or weights missing")
    actual = set()
    for parent, dirs, files in os.walk(directory, followlinks=False, onerror=lambda exc: (_ for _ in ()).throw(exc)):
        for name in dirs + files:
            path = Path(parent) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError("invalid model file type")
            if stat.S_ISREG(info.st_mode):
                relative = path.relative_to(directory).as_posix()
                if relative != MANIFEST:
                    actual.add(relative)
    if actual != expected.keys():
        raise ValueError("model files do not match manifest")
    for name, (size, kind, digest) in expected.items():
        hasher = hashlib.sha256() if kind == "sha256" else hashlib.sha1()
        if kind == "git-blob-sha1":
            hasher.update(f"blob {size}\0".encode())
        count = 0
        with (directory / name).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                count += len(chunk)
                hasher.update(chunk)
        if count != size or hasher.hexdigest() != digest:
            raise ValueError("model file digest mismatch")
    if expected["config.json"][0] > 10_000_000:
        raise ValueError("model configuration too large")
    if not isinstance(json.loads((directory / "config.json").read_bytes()), dict):
        raise ValueError("invalid model configuration")
    return manifest
