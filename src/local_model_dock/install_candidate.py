"""Interactive confirmation for downloading one explicitly selected candidate."""

import re
import sys
from pathlib import Path

from . import model_download
from .selection import assess_discovered, resource_estimate
from .storage import Confirmation, inspect_storage


def install_candidate(model, *, backend, root, cache_bytes, memory_bytes,
                      budget_bytes=20_000_000_000, input_stream=None,
                      output_stream=None):
    """Display an unverified candidate and download only after exact consent."""
    input_stream = sys.stdin if input_stream is None else input_stream
    output_stream = sys.stdout if output_stream is None else output_stream
    if not input_stream.isatty() or not output_stream.isatty():
        raise PermissionError("interactive terminal required")
    if (not isinstance(model.name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", model.name)
            or not isinstance(model.revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", model.revision)):
        raise ValueError("valid repository and immutable commit required")
    if type(model.disk_bytes) is not int or model.disk_bytes <= 0:
        raise ValueError("complete repository download size is unknown")

    root = Path(root)
    project_root = root.parent
    status = inspect_storage(root, model.disk_bytes, budget_bytes, cache_bytes,
                             project_root=project_root)
    artifact, required_memory, required_disk = resource_estimate(model, backend)
    _, pending = assess_discovered([model], memory_bytes, status.free_bytes, (), backend)
    print(f"Source: https://huggingface.co/{model.name}", file=output_stream)
    print(f"Revision: {model.revision}", file=output_stream)
    print(f"Repository size: {model.disk_bytes} bytes", file=output_stream)
    print(f"License: {model.license!r} (eligibility requires review)", file=output_stream)
    print(f"Backend: {backend!r}; artifact: {artifact!r}; estimated memory: {required_memory}; estimated disk: {required_disk}", file=output_stream)
    print(f"Storage: used {status.used_bytes}, projected {status.projected_bytes}, budget {budget_bytes}, free {status.free_bytes} bytes", file=output_stream)
    print("Not verified or formally recommended; no automatic activation.", file=output_stream)
    print("Risks: " + ", ".join(pending[0][1]), file=output_stream)
    if status.free_bytes < model.disk_bytes:
        raise OSError("insufficient disk space")

    identity = f"{model.name}@{model.revision}"
    print(f"Type {identity} to download:", file=output_stream, flush=True)
    if input_stream.readline().rstrip("\r\n") != identity:
        raise PermissionError("candidate and revision confirmation required")
    if status.over_budget:
        confirmation = f"over-budget {identity} {status.projected_bytes}"
        print(f"Budget exceeded. Type {confirmation} to confirm:", file=output_stream, flush=True)
        if input_stream.readline().rstrip("\r\n") != confirmation:
            raise PermissionError("separate over-budget confirmation required")
    return model_download.download(
        root=root, repo=model.name, revision=model.revision,
        confirmation=Confirmation(model.name, model.revision),
        cache_bytes=cache_bytes, budget_bytes=budget_bytes,
        project_root=project_root,
        max_confirmed_projected_bytes=status.projected_bytes if status.over_budget else None,
    )
