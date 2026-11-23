from dataclasses import dataclass


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


def select(candidates: list[Candidate], memory: int | None, disk: int | None,
           families: tuple[str, ...]) -> list[Candidate]:
    if memory is None or disk is None:
        return []
    eligible = [
        c for c in candidates
        if c.moe is True and c.compatible is True and c.accessible is True
        and c.licensed is True and c.memory_bytes is not None
        and 0 < c.memory_bytes <= memory and c.disk_bytes is not None
        and 0 < c.disk_bytes <= disk and c.model_revision
        and c.backend_version and c.tested_revision == c.model_revision
        and c.tested_backend == c.backend_version and c.tool_test_passed is True
    ]
    return sorted(eligible, key=lambda c: (
        families.index(c.family) if c.family in families else len(families),
        -c.coding_score,
    ))
