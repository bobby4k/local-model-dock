"""Read-only Hugging Face discovery; metadata is not a quality verdict."""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from urllib.parse import urlsplit

from .validation import detect_moe


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open(url):
    return urllib.request.build_opener(NoRedirect()).open(url, timeout=15)


@dataclass(frozen=True)
class DiscoveryCandidate:
    name: str
    revision: str | None
    license: str | None
    downloads: int | None
    moe: bool | None = None
    verified: bool = False
    formats: tuple[str, ...] = ()
    accessible: bool | None = None
    disk_bytes: int | None = None
    gguf_files: tuple[tuple[str, int], ...] = ()
    family: str | None = None
    version: tuple[int, ...] | None = None
    total_parameters: int | None = None


def inspect_candidate(model: DiscoveryCandidate) -> DiscoveryCandidate:
    """Read bounded metadata pinned to a Git commit, never model weights."""
    if (not isinstance(model.revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", model.revision)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", model.name)):
        return model
    base = "https://huggingface.co"
    try:
        with _open(f"{base}/{model.name}/raw/{model.revision}/config.json") as response:
            data = response.read(1_000_001)
            if len(data) > 1_000_000:
                raise ValueError("Model config response too large")
            config = json.loads(data)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        config = None
    with _open(f"{base}/api/models/{model.name}/tree/{model.revision}?recursive=true&expand=true") as response:
        data = response.read(2_000_001)
        if len(data) > 2_000_000 or response.headers.get("Link"):
            raise ValueError("Incomplete model file listing")
        files = json.loads(data)
    if not isinstance(files, list):
        raise ValueError("Invalid model file listing")
    regular = [f for f in files if isinstance(f, dict) and f.get("type", "file") == "file"]
    sizes_known = bool(regular) and all(type(f.get("size")) is int and f["size"] >= 0 for f in regular)
    paths = [f.get("path") for f in regular]
    quant = config.get("quantization") if isinstance(config, dict) else None
    mlx = (isinstance(quant, dict) and type(quant.get("bits")) is int
           and quant["bits"] in (2, 3, 4, 6, 8)
           and quant.get("quant_method") in (None, "mlx")
           and any(isinstance(p, str) and p.endswith(".safetensors") for p in paths))
    # Multiple GGUF variants need individual file identities; do not sum them as one model.
    gguf = tuple((f["path"], f["size"]) for f in regular
                 if isinstance(f.get("path"), str) and f["path"].endswith(".gguf")
                 and type(f.get("size")) is int and f["size"] > 0)
    formats = (("mlx",) if mlx else ()) + (("gguf",) if gguf else ())
    model_type = config.get("model_type") if isinstance(config, dict) else None
    match = re.fullmatch(r"(qwen|deepseek)(?:_?v)?(\d+)(?:_(\d+))?(?:_moe)?", model_type or "") if isinstance(model_type, str) else None
    family = match.group(1) if match else ("other" if model_type == "mixtral" else None)
    version = tuple(int(part) for part in match.groups()[1:] if part is not None) if match else None
    return replace(model, moe=detect_moe(config), formats=formats, gguf_files=gguf,
                   family=family, version=version,
                   total_parameters=model.total_parameters if isinstance(config, dict)
                   and not any(key in config for key in ("quantization", "quantization_config")) else None,
                   disk_bytes=sum(f["size"] for f in regular) if mlx and sizes_known else None)


def discover_fallback(repo: str) -> DiscoveryCandidate:
    """Resolve one configured repository to its current immutable revision."""
    if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo):
        raise ValueError("Invalid fallback repository")
    with _open(f"https://huggingface.co/api/models/{repo}") as response:
        data = response.read(2_000_001)
        if len(data) > 2_000_000:
            raise ValueError("Model response too large")
        item = json.loads(data)
    if (not isinstance(item, dict) or item.get("id") != repo
            or not isinstance(item.get("sha"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", item["sha"])):
        raise ValueError("Fallback identity or revision unavailable")
    card = item.get("cardData")
    license_name = card.get("license") if isinstance(card, dict) else None
    private, gated = item.get("private"), item.get("gated")
    accessible = (False if private is True or gated in ("auto", "manual") or gated is True
                  else True if private is False and gated is False else None)
    tensors = item.get("safetensors")
    total = tensors.get("total") if isinstance(tensors, dict) else None
    total = total if type(total) is int and total > 0 else None
    return DiscoveryCandidate(repo, item["sha"], license_name, item.get("downloads"),
                              accessible=accessible, total_parameters=total)


def discover(max_pages: int = 3, *, _url: str | None = None) -> list[DiscoveryCandidate]:
    url = _url or ("https://huggingface.co/api/models?filter=text-generation"
           "&limit=10&expand[]=sha&expand[]=cardData&expand[]=downloads"
           "&expand[]=config&expand[]=siblings&expand[]=private&expand[]=gated&expand[]=safetensors")
    results = []
    seen = set()
    for _ in range(max_pages):
        parts = urlsplit(url)
        if (parts.scheme != "https" or parts.netloc != "huggingface.co"
                or parts.path != "/api/models" or url in seen):
            raise ValueError("Invalid pagination URL")
        seen.add(url)
        with _open(url) as response:
            body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise ValueError("Model list response too large")
            data = json.loads(body)
            link = response.headers.get("Link", "")
        if not isinstance(data, list):
            raise ValueError("Invalid model list response")
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            card = item.get("cardData")
            license_name = card.get("license") if isinstance(card, dict) else None
            siblings = item.get("siblings")
            names = [entry.get("rfilename") for entry in siblings if isinstance(entry, dict)] if isinstance(siblings, list) else []
            formats = tuple(fmt for fmt in ("gguf", "safetensors")
                            if any(isinstance(name, str) and name.endswith("." + fmt) for name in names))
            private, gated = item.get("private"), item.get("gated")
            tensors = item.get("safetensors")
            total = tensors.get("total") if isinstance(tensors, dict) else None
            total = total if type(total) is int and total > 0 and re.fullmatch(r"[0-9a-f]{40}", item.get("sha") or "") else None
            accessible = (False if private is True or gated in ("auto", "manual") or gated is True
                          else True if private is False and gated is False else None)
            results.append(DiscoveryCandidate(item["id"], item.get("sha"), license_name,
                                              item.get("downloads"), detect_moe(item.get("config")),
                                              formats=formats, accessible=accessible, total_parameters=total))
        match = re.search(r'<([^>]+)>\s*;\s*rel="next"', link)
        if not match:
            break
        url = match.group(1)
    return results


def discover_targeted(max_pages: int = 3) -> list[DiscoveryCandidate]:
    """Search format-specific repositories, preserving both search order and identity."""
    results = {}
    # shortcut: naming-based searches miss other compact architectures; broaden discovery when metadata filters allow it.
    for query in ("Qwen3.5", "DeepSeek", "MoE-A2", "mlx", "gguf"):
        url = (f"https://huggingface.co/api/models?search={query}&filter=text-generation"
               "&limit=50&expand[]=sha&expand[]=cardData&expand[]=downloads"
               "&expand[]=config&expand[]=siblings&expand[]=private&expand[]=gated&expand[]=safetensors")
        for model in discover(max_pages, _url=url):
            results.setdefault(model.name, model)
    return list(results.values())
