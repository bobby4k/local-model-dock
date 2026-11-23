from dataclasses import dataclass
import re


BLOCKERS = {"MoE not confirmed", "access conditions", "license eligibility",
            "immutable revision", "backend format", "available memory", "available disk",
            "insufficient memory estimate", "insufficient disk estimate", "model family, version or total parameters"}


def meets_model_limits(family, version, total_parameters):
    if not family or type(total_parameters) is not int or total_parameters < 9_000_000_000:
        return False
    minimum = {"qwen": (3, 5), "deepseek": (4,)}.get(family.lower())
    return minimum is None or (isinstance(version, tuple) and bool(version)
                               and all(type(v) is int and v >= 0 for v in version)
                               and version >= minimum)


def resource_estimate(model, backend, *, quantization=None):
    artifacts = [(name, size) for name, size in model.gguf_files
                 if re.search(r"(?:^|[-_.])[iq][1-8](?:_[a-z0-9]+)*\.gguf$", name, re.I)
                 and not re.search(r"(?:^|/)(?:mmproj|projector|vision|clip)[-_.]", name, re.I)]
    if backend == "ollama" and quantization is not None:
        artifacts = [item for item in artifacts if (match := re.search(
            r"(?:^|[-_.])((?:i[1-8]-)?[iq][1-8](?:_[a-z0-9]+)*)\.gguf$", item[0], re.I))
            and match.group(1).lower() == quantization.lower()]
        if len(artifacts) != 1:
            raise ValueError(f"Fallback quantization {quantization!r} not found or ambiguous")
    artifact = min(artifacts, key=lambda item: item[1], default=None) if backend == "ollama" else None
    size = artifact[1] if artifact else (
        model.disk_bytes if backend == "omlx" and "mlx" in model.formats else None)
    if not size or size <= 0:
        return None, None, None
    # shortcut: estimates exclude architecture/context-specific runtime allocations; calibrate on the target backend.
    return artifact[0] if artifact else "repository", size * 3 // 2 + 2 * 1024**3, size * 2 + 256 * 1024**2


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    moe: bool | None
    compatible: bool | None
    accessible: bool | None
    licensed: bool | None
    memory_bytes: int | None
    disk_bytes: int | None
    model_revision: str
    backend_version: str
    tested_revision: str
    tested_backend: str
    tool_test_passed: bool
    coding_score: float
    backend: str = ""
    tested_backend_name: str = ""
    version: tuple[int, ...] | None = None
    total_parameters: int | None = None


def select(candidates: list[Candidate], memory: int | None, disk: int | None,
           families: tuple[str, ...], *, fallback_repo: str | None = None) -> list[Candidate]:
    if memory is None or disk is None:
        return []
    eligible = [
        c for c in candidates
        if meets_model_limits(c.family, c.version, c.total_parameters)
        and (c.moe is True or (fallback_repo is not None and c.name == fallback_repo))
        and c.compatible is True and c.accessible is True
        and c.licensed is True and c.memory_bytes is not None
        and 0 < c.memory_bytes <= memory and c.disk_bytes is not None
        and 0 < c.disk_bytes <= disk
        and re.fullmatch(r"(?:[0-9a-f]{40}|(?:sha256:)?[0-9a-f]{64})", c.model_revision)
        and c.backend_version and c.tested_revision == c.model_revision
        and c.tested_backend == c.backend_version and c.tool_test_passed is True
        and c.backend in ("omlx", "ollama") and c.tested_backend_name == c.backend
    ]
    return sorted(eligible, key=lambda c: (
        c.name == fallback_repo,
        families.index(c.family) if c.family in families else len(families),
        -c.coding_score,
    ))


def assess_verified(models, memory, disk, families, backend, backend_version,
                    allowed_licenses, check_identity, *, fallback_repo=None, fallback_quantization=None):
    """Select models only when a trusted checker verifies this entire identity.

    check_identity(repo, revision, backend, backend_version) must return True
    only after the supervisor has checked that exact identity. A check of a
    model name alone does not satisfy this contract.
    """
    if backend not in ("omlx", "ollama") or not isinstance(backend_version, str) or not backend_version:
        return []
    candidates = []
    for model in models:
        if (not isinstance(model.name, str) or not model.name
                or not isinstance(model.revision, str)
                or not re.fullmatch(r"[0-9a-f]{40}", model.revision)
                or not isinstance(model.license, str)
                or not model.license or model.license not in allowed_licenses):
            continue
        _, required_memory, required_disk = resource_estimate(
            model, backend, quantization=fallback_quantization if model.name == fallback_repo else None)
        if (not meets_model_limits(model.family, model.version, model.total_parameters)
                or (model.moe is not True and (fallback_repo is None or model.name != fallback_repo))
                or model.accessible is not True
                or required_memory is None or memory is None or required_memory > memory
                or required_disk is None or disk is None or required_disk > disk):
            continue
        try:
            verified = check_identity(model.name, model.revision, backend, backend_version) is True
        except Exception:
            verified = False
        if not verified:
            continue
        family = model.family
        candidates.append(Candidate(model.name, family, model.moe, True, model.accessible,
                                    True, required_memory, required_disk, model.revision,
                                    backend_version, model.revision, backend_version, True, 0,
                                    backend, backend, model.version, model.total_parameters))
    return select(candidates, memory, disk, families, fallback_repo=fallback_repo)


def assess_discovered(models, memory: int | None, disk: int | None,
                      families: tuple[str, ...], backend: str | None = None, *, fallback_repo=None,
                      fallback_quantization=None):
    """Assess discovery metadata without treating model-card claims as evidence."""
    if backend is not None:
        def assess(model):
            unknowns = []
            if not meets_model_limits(model.family, model.version, model.total_parameters):
                unknowns.append("model family, version or total parameters")
            _, required_memory, required_disk = resource_estimate(
                model, backend, quantization=fallback_quantization if model.name == fallback_repo else None)
            if model.moe is not True and (fallback_repo is None or model.name != fallback_repo):
                unknowns.append("MoE not confirmed")
            if model.accessible is not True:
                unknowns.append("access conditions")
            if not model.license:
                unknowns.append("license eligibility")
            else:
                unknowns.append("license eligibility requires review")
            if not re.fullmatch(r"[0-9a-f]{40}", model.revision or ""):
                unknowns.append("immutable revision")
            if required_memory is None:
                unknowns.append("backend format")
                unknowns.extend(("memory estimate", "disk estimate"))
            else:
                if memory is None:
                    unknowns.append("available memory")
                elif memory < required_memory:
                    unknowns.append("insufficient memory estimate")
                if disk is None:
                    unknowns.append("available disk")
                elif disk < required_disk:
                    unknowns.append("insufficient disk estimate")
                unknowns.append("runtime and context memory not verified")
            unknowns.extend(("backend architecture compatibility", "tool-call verification"))
            return model, tuple(unknowns)

        pending = [assess(model) for model in models]
        def key(item):
            model, unknowns = item
            family = next((i for i, name in enumerate(families)
                           if name.lower() == (model.family or "").lower()), len(families))
            return bool(BLOCKERS.intersection(unknowns)), model.name == fallback_repo, family
        return [], sorted(pending, key=key)
    candidates = [
        Candidate(model.name, "", None, None, None, None, None, None,
                  model.revision or "", "", "", "", False, 0)
        for model in models
    ]
    recommendations = select(candidates, memory, disk, families)
    unknowns = ("MoE", "backend compatibility", "access conditions",
                "license eligibility", "memory estimate", "disk estimate",
                "tool-call verification")
    return recommendations, [
        (model, tuple(item for item in unknowns if item != "MoE" or model.moe is None))
        for model in models
    ]
