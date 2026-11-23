"""Project-owned file storage; downloaded files are not model-validated."""

import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable


@dataclass(frozen=True)
class Confirmation:
    name: str
    revision: str


@dataclass(frozen=True)
class StorageStatus:
    used_bytes: int
    projected_bytes: int
    free_bytes: int
    over_budget: bool


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    used_bytes: int
    over_budget: bool


def inspect_storage(root: Path, size_bytes: int, budget_bytes: int, cache_bytes: int,
                    *, project_root: Path | None = None) -> StorageStatus:
    """Count project files and caller-observed cache; fail closed on unknown usage."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('storage root must be an existing non-symlink directory')
    if any(type(n) is not int or n < 0 for n in (size_bytes, budget_bytes, cache_bytes)) or not size_bytes:
        raise ValueError('sizes must be known non-negative integers and download size positive')
    if project_root is not None:
        project_root = Path(os.path.abspath(project_root))
        root = Path(os.path.abspath(root))
        if (root != project_root / 'models' or project_root.is_symlink()
                or not project_root.is_dir() or cache_bytes != 0):
            raise ValueError('project storage must use models and a measured cache')
    scan_root = project_root if project_root is not None else root
    used = cache_bytes
    seen = set()
    device = scan_root.stat().st_dev
    for directory, dirs, files in os.walk(scan_root, followlinks=False, onerror=lambda e: (_ for _ in ()).throw(e)):
        for name in dirs + files:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise ValueError('symlinks in storage are not supported')
            if info.st_dev != device:
                raise ValueError('storage crossing filesystems is not supported')
            if name in files:
                if (project_root is not None and path == project_root / 'control.sock'
                        and stat.S_ISSOCK(info.st_mode) and info.st_uid == os.getuid()):
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError('storage contains a non-regular file')
                identity = (info.st_dev, info.st_ino)
                if identity not in seen:
                    used += info.st_size
                    seen.add(identity)
    free = shutil.disk_usage(root).free
    return StorageStatus(used, used + size_bytes, free, used + size_bytes > budget_bytes)


def download(*, root: Path, name: str, revision: str, size_bytes: int,
             budget_bytes: int, cache_bytes: int, confirmation: Confirmation | None,
             write: Callable[[BinaryIO], object]) -> DownloadResult:
    """Write one confirmed candidate atomically without replacing existing files.

    Call inspect_storage before confirmation to show projected budget usage. The
    writer must fetch exactly the confirmed revision and write exactly size_bytes.
    """
    if not name or name in ('.', '..') or '/' in name or '\\' in name or '\0' in name:
        raise ValueError('name must be a single safe file name')
    if not revision or confirmation != Confirmation(name, revision):
        raise PermissionError('explicit confirmation for this candidate is required')
    root = Path(root)
    status = inspect_storage(root, size_bytes, budget_bytes, cache_bytes)
    if status.free_bytes < size_bytes:
        raise OSError('insufficient disk space; free space or adjust storage')
    target = root / name
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    fd, temporary = tempfile.mkstemp(prefix='.download-', dir=root)
    try:
        with os.fdopen(fd, 'wb') as stream:
            write(stream)
            stream.flush()
            if os.fstat(stream.fileno()).st_size != size_bytes:
                raise ValueError('download size does not match estimate')
        os.link(temporary, target, follow_symlinks=False)
    finally:
        os.unlink(temporary)
    actual = inspect_storage(root, size_bytes, budget_bytes, cache_bytes).used_bytes
    return DownloadResult(target, actual, actual > budget_bytes)
