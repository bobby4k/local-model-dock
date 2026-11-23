"""Confirmed, project-private Hugging Face repository downloads."""

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlsplit

from .storage import Confirmation, inspect_storage


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    @staticmethod
    def allowed(url):
        parts = urlsplit(url)
        return (parts.scheme == "https" and parts.netloc in {
            "huggingface.co", "cdn-lfs.huggingface.co",
            "cas-bridge.xethub.hf.co", "transfer.xethub.hf.co",
        })

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not self.allowed(newurl):
            raise ValueError("unsafe download redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(url):
    if not SafeRedirect.allowed(url):
        raise ValueError("unsafe download URL")
    return urllib.request.build_opener(SafeRedirect()).open(url, timeout=30)


def _publish(source, target):
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = libc.renamex_np
        args = (os.fsencode(source), os.fsencode(target), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        args = (-100, os.fsencode(source), -100, os.fsencode(target), 1)  # RENAME_NOREPLACE
    else:
        raise OSError("atomic no-replace directory publication unavailable")
    if function(*args) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(target)
        raise OSError(error, os.strerror(error))


def download(*, root, repo, revision, confirmation, cache_bytes,
             budget_bytes=20_000_000_000, max_confirmed_projected_bytes=None,
             project_root=None):
    """Download a confirmed immutable public repository; never authenticate.

    The caller must present the projected usage before confirming its upper
    limit. Repository metadata and project storage are checked again.
    """
    if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo):
        raise ValueError("invalid repository")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("an immutable commit revision is required")
    if confirmation != Confirmation(repo, revision):
        raise PermissionError("explicit repository and revision confirmation required")
    if (max_confirmed_projected_bytes is not None
            and (type(max_confirmed_projected_bytes) is not int
                 or max_confirmed_projected_bytes < 0)):
        raise ValueError("confirmed projected usage must be a non-negative integer")
    root = Path(root)
    # Check the root before contacting the network.
    inspect_storage(root, 1, budget_bytes, cache_bytes, project_root=project_root)
    api = f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true&expand=true"
    with _open(api) as response:
        if response.headers.get("Link"):
            raise ValueError("paginated repository metadata is not supported")
        data = response.read(10_000_001)
    if len(data) > 10_000_000:
        raise ValueError("repository metadata too large")
    entries = json.loads(data)
    if not isinstance(entries, list):
        raise ValueError("invalid repository metadata")
    files = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid repository entry")
        if entry.get("type") == "directory":
            continue
        name, size = entry.get("path"), entry.get("size")
        if (entry.get("type") != "file" or not isinstance(name, str)
                or "\\" in name or "\0" in name or not name
                or name == ".local-model-dock-manifest.json"
                or any(part in ("", ".", "..") for part in name.split("/"))
                or PurePosixPath(name).is_absolute() or name in seen
                or type(size) is not int or size < 0):
            raise ValueError("invalid repository file")
        seen.add(name)
        lfs = entry.get("lfs")
        digest = lfs.get("oid") if isinstance(lfs, dict) else None
        if lfs is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("invalid LFS digest")
        if lfs is None:
            digest = entry.get("oid")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{40}", digest):
                raise ValueError("invalid git blob digest")
        files.append((name, size, digest))
    total = sum(size for _, size, _ in files)
    if not total:
        raise ValueError("repository size is unknown or empty")
    status = inspect_storage(root, total, budget_bytes, cache_bytes, project_root=project_root)
    if status.free_bytes < total:
        raise OSError("insufficient disk space")
    if (status.over_budget and (max_confirmed_projected_bytes is None
            or status.projected_bytes > max_confirmed_projected_bytes)):
        raise PermissionError(f"projected usage {status.projected_bytes} exceeds budget {budget_bytes} or confirmed limit; explicit confirmation required")
    target = root / (repo.replace("/", "--") + "--" + revision)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    temporary = Path(tempfile.mkdtemp(prefix=".download-", dir=root))
    try:
        for name, size, digest in files:
            destination = temporary.joinpath(*name.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(name, safe='/')}"
            hasher = hashlib.sha256() if len(digest) == 64 else hashlib.sha1()
            if len(digest) == 40:
                hasher.update(f"blob {size}\0".encode())
            count = 0
            with _open(url) as response, destination.open("xb") as output:
                while True:
                    chunk = response.read(min(1024 * 1024, size - count + 1))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > size:
                        raise ValueError("download exceeds trusted size")
                    hasher.update(chunk)
                    output.write(chunk)
            if count != size or hasher.hexdigest() != digest:
                raise ValueError("download size or digest mismatch")
        manifest = {
            "version": 1, "repo": repo, "revision": revision,
            "files": [{"path": name, "size": size,
                       "digest_type": "sha256" if len(digest) == 64 else "git-blob-sha1",
                       "digest": digest} for name, size, digest in files],
        }
        (temporary / ".local-model-dock-manifest.json").write_text(json.dumps(manifest))
        _publish(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target
