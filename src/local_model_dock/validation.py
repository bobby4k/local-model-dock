"""Conservative architecture detection and an injectable tool-call probe."""

import json
import re
from dataclasses import dataclass
from typing import Callable


def detect_moe(config: object) -> bool | None:
    if not isinstance(config, dict) or config.get("model_type") not in ("mixtral", "qwen2_moe", "qwen3_moe", "qwen3_5_moe"):
        return None
    experts = config.get("num_local_experts" if config["model_type"] == "mixtral" else "num_experts")
    active = config.get("num_experts_per_tok")
    if type(experts) is int and type(active) is int and 1 <= active <= experts and experts > 1:
        return True
    return None


@dataclass(frozen=True)
class ToolTestRecord:
    model_revision: str
    backend: str
    backend_version: str
    passed: bool

    def matches(self, model_revision: str, backend: str, backend_version: str) -> bool:
        return (self.passed and self.model_revision == model_revision
                and self.backend == backend and self.backend_version == backend_version)


def validate_tool_call(
    request: Callable[[dict], object], model_revision: str, backend: str, backend_version: str
) -> ToolTestRecord:
    passed = False
    # shortcut: only full Git SHA-1 and SHA-256 revisions are recognized; add verified digest formats as needed.
    identity_ok = (isinstance(model_revision, str)
                   and re.fullmatch(r"(?:[0-9a-f]{40}|(?:sha256:)?[0-9a-f]{64})", model_revision)
                   and isinstance(backend, str) and bool(backend.strip())
                   and isinstance(backend_version, str)
                   and re.fullmatch(r"[0-9][0-9A-Za-z.+_-]*", backend_version))
    if identity_ok:
        probe = {
            "model": model_revision,
            "messages": [{"role": "user", "content": 'Call lookup with query "dock".'}],
            "tools": [{"type": "function", "function": {
                "name": "lookup", "description": "Look up a query",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                               "required": ["query"], "additionalProperties": False},
            }}],
            "tool_choice": "required",
        }
        try:
            calls = request(probe)["choices"][0]["message"]["tool_calls"]
            if isinstance(calls, list) and len(calls) == 1 and calls[0]["type"] == "function":
                function = calls[0]["function"]
                def reject_constant(value):
                    raise ValueError(value)
                def unique_object(pairs):
                    result = {}
                    for key, value in pairs:
                        if key in result:
                            raise ValueError("duplicate key")
                        result[key] = value
                    return result
                arguments = json.loads(function["arguments"], parse_constant=reject_constant,
                                       object_pairs_hook=unique_object)
                passed = function["name"] == "lookup" and arguments == {"query": "dock"}
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            pass
    return ToolTestRecord(model_revision, backend, backend_version, passed)
