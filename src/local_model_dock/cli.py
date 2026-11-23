import argparse
import json
import re
import socket
import stat
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace


def _control(action, model):
    from . import supervisor
    path = supervisor.private_root() / supervisor.SOCKET
    info = path.lstat()
    import os
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError("control path is not an owned socket")
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(150)
        client.connect(str(path))
        client.sendall(json.dumps({"action": action, "model": model}).encode() + b"\n")
        response = b""
        while not response.endswith(b"\n") and len(response) <= 4096:
            chunk = client.recv(4097 - len(response))
            if not chunk:
                break
            response += chunk
    if not response.endswith(b"\n") or len(response) > 4096:
        raise RuntimeError("invalid supervisor response")
    result = json.loads(response)
    if not isinstance(result, dict) or result.get("verified") is not True:
        raise RuntimeError(f"{action} failed" + (" (supervisor busy)" if isinstance(result, dict) and result.get("busy") else ""))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(prog="local-model-dock")
    parser.add_argument("command", choices=("start", "update", "stop", "restart"))
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--install", metavar="REPO", help="interactively download and verify a discovered candidate")
    operation.add_argument("--verify", metavar="MODEL", help="retry verification of a downloaded model directory")
    operation.add_argument("--enable", metavar="MODEL", help="enable a downloaded model with a valid verification record")
    parser.add_argument("--activate", action="store_true", help="enable the candidate after verification")
    args = parser.parse_args()
    if (args.install or args.verify or args.enable or args.activate) and args.command != "update":
        parser.error("model options are only valid with update")
    if args.activate and not (args.install or args.verify):
        parser.error("--activate requires --install or --verify")
    if args.verify or args.enable:
        model = args.verify or args.enable
        if model == "local-dock-initial":
            try:
                if args.verify:
                    _control("verify", model)
                    _control("check", model)
                    print(f"Verified: {model}")
                if args.enable or args.activate:
                    _control("enable", model)
                    print(f"Enabled: {model}")
                return 0
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"update: {exc}", file=sys.stderr)
                return 1
        print("update: trusted policy metadata unavailable for downloaded models", file=sys.stderr)
        return 1
    if args.command == "update":
        from .discovery import discover_targeted, discover_fallback, inspect_candidate
        from .hardware import print_hardware, available_memory, available_disk, detect_backend
        from .config import load_config
        from .selection import assess_discovered, assess_verified, meets_model_limits, resource_estimate, BLOCKERS
        print_hardware()
        try:
            config = load_config()
            backend = detect_backend()
            try:
                discovered = discover_targeted()
                search_failed = False
            except (OSError, ValueError, urllib.error.URLError) as exc:
                print(f"Discovery failed: {exc}", file=sys.stderr)
                discovered = []
                search_failed = True
            try:
                fallback = discover_fallback(config.fallback_repo)
            except (OSError, ValueError, urllib.error.URLError) as exc:
                fallback = None
                print(f"Fallback discovery unavailable: {exc}", file=sys.stderr)
            if search_failed and fallback is None:
                return 1
            discovered = [model for model in discovered if model.name != config.fallback_repo]
            # shortcut: inspect at most 24 search results; expand bounded discovery for larger catalogs.
            def priority(model):
                platform_match = ("mlx" in model.name.lower()) if backend == "omlx" else ("gguf" in model.name.lower())
                family = next((i for i, name in enumerate(config.families)
                               if name.lower() in model.name.lower()), len(config.families))
                return not platform_match, family, model.moe is not True
            discovered.sort(key=priority)
            def inspect(model):
                try:
                    return inspect_candidate(model)
                except (OSError, ValueError, urllib.error.URLError):
                    return replace(model, moe=None, formats=(), disk_bytes=None, gguf_files=())
            with ThreadPoolExecutor(max_workers=8) as pool:
                candidates = list(pool.map(inspect, discovered[:24]))
            if fallback is not None:
                candidates.append(inspect(fallback))
            memory, disk = available_memory(), available_disk(".")
            recommendations, pending = assess_discovered(candidates, memory, disk, config.families,
                                                          backend=backend, fallback_repo=config.fallback_repo,
                                                          fallback_quantization=config.fallback_quantization)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            print(f"Discovery failed: {exc}", file=sys.stderr)
            return 1
        if not args.install:
            allowed_licenses = {"apache-2.0", "mit"}
            for candidate in candidates:
                is_fallback = candidate.name == config.fallback_repo
                _, required_memory, required_disk = resource_estimate(
                    candidate, backend, quantization=config.fallback_quantization if is_fallback else None)
                if (candidate.license not in allowed_licenses or (candidate.moe is not True and not is_fallback)
                        or candidate.accessible is not True
                        or not isinstance(candidate.name, str)
                        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", candidate.name)
                        or not isinstance(candidate.revision, str)
                        or not re.fullmatch(r"[0-9a-f]{40}", candidate.revision)
                        or required_memory is None or memory is None or required_memory > memory
                        or required_disk is None or disk is None or required_disk > disk):
                    continue
                if is_fallback:
                    # shortcut: GGUF headers are not verified per artifact; enable after pinned metadata extraction.
                    if backend == "ollama":
                        continue
                    from . import supervisor
                    from .storage import inspect_storage
                    try:
                        root = supervisor.private_root() / "models"
                        size = next((size for name, size in candidate.gguf_files
                                     if name == resource_estimate(candidate, backend)[0]), candidate.disk_bytes)
                        storage = inspect_storage(root, size, int(config.budget_gb * 1_000_000_000),
                                                  0, project_root=root.parent)
                        if storage.over_budget:
                            continue
                    except (OSError, ValueError, TypeError):
                        continue
                model = candidate.name.replace("/", "--") + "--" + candidate.revision
                try:
                    result = _control("check", model)
                    identity = result["identity"]
                    version = identity["backend_version"]
                    if (result.get("verified") is not True
                            or not isinstance(version, str)
                            or not re.fullmatch(r"[0-9][0-9A-Za-z.+_-]*", version)
                            or identity != {"repo": candidate.name, "revision": candidate.revision,
                                            "backend": backend, "backend_version": version}):
                        continue
                    def check_identity(repo, revision, backend_name, backend_version):
                        checked = _control("check", model)
                        return (checked.get("verified") is True
                                and checked.get("identity") == {
                                    "repo": repo, "revision": revision, "backend": backend_name,
                                    "backend_version": backend_version})
                    recommendations.extend(assess_verified(
                        [candidate], memory, disk, config.families, backend, version,
                         allowed_licenses, check_identity, fallback_repo=config.fallback_repo,
                         fallback_quantization=config.fallback_quantization))
                except (OSError, ValueError, RuntimeError, KeyError, TypeError):
                    continue
            recommendations.sort(key=lambda item: (
                item.name == config.fallback_repo,
                config.families.index(item.family) if item.family in config.families else len(config.families),
                -item.coding_score))
            recommended_names = {item.name for item in recommendations}
            pending = [(model, unknowns) for model, unknowns in pending if model.name not in recommended_names]
        if recommendations:
            print(f"Recommended: {recommendations[0].name}")
        else:
            print("No verified recommendation: model and hardware evidence is incomplete.")
        print("Unverified discovery candidates (not recommendations):")
        if pending and all(BLOCKERS.intersection(unknowns) for _, unknowns in pending):
            print("No candidate passes the preliminary resource and evidence checks.")
        for candidate, unknowns in pending:
            moe = "yes" if candidate.moe is True else "no" if candidate.moe is False else "unknown"
            artifact, ram, disk = resource_estimate(
                candidate, backend, quantization=config.fallback_quantization
                if candidate.name == config.fallback_repo else None)
            estimate = (f"{ram / 1024**3:.2f} GiB RAM, {disk / 1024**3:.2f} GiB disk"
                        if ram is not None else "unknown")
            status = "not suitable for preselection" if BLOCKERS.intersection(unknowns) else "preliminary candidate"
            print(f"{candidate.name} | revision: {candidate.revision or 'unknown'}"
                  f" | license: {candidate.license or 'unknown'}"
                  f" | MoE: {moe} | unverified | {status}"
                  f" | format: {', '.join(candidate.formats) or 'unknown'}"
                  f" | artifact: {artifact or 'unknown'}"
                  f" | resource estimate: {estimate}"
                  f" | Unknown: {', '.join(unknowns)}")
            if candidate.name == config.fallback_repo:
                print("Fallback quantizations (unverified): " + (
                    ", ".join(f"{name} ({size} bytes)" for name, size in candidate.gguf_files)
                    or "none discovered; GGUF metadata and budget require verification"))
                print("Fallback budget and GGUF metadata require verification before recommendation.")
        if not candidates:
            print("No candidates found.")
        if args.install:
            try:
                if not sys.stdin.isatty() or not sys.stdout.isatty():
                    raise PermissionError("interactive terminal required")
                if backend != "omlx":
                    raise ValueError("verification is currently supported only on oMLX")
                matches = [model for model in candidates if model.name == args.install]
                if len(matches) != 1:
                    raise ValueError("candidate must match exactly one inspected discovery result")
                if not meets_model_limits(matches[0].family, matches[0].version, matches[0].total_parameters):
                    raise ValueError("model family, version or total parameters do not meet policy or are unknown")
                from . import supervisor
                from .install_candidate import install_candidate
                root = supervisor.private_root() / "models"
                model = install_candidate(matches[0], backend=backend, root=root,
                                          cache_bytes=0, memory_bytes=available_memory(),
                                          budget_bytes=int(config.budget_gb * 1_000_000_000))
                print(f"Downloaded model: {model.name}")
                _control("verify", model.name)
                _control("check", model.name)
                print(f"Verified: {model.name}")
                if args.activate:
                    _control("enable", model.name)
                    print(f"Enabled: {model.name}")
            except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
                print(f"update: {exc}", file=sys.stderr)
                return 1
        return 0
    from . import supervisor
    try:
        if args.command in ("stop", "restart"):
            supervisor.stop()
        if args.command in ("start", "restart"):
            supervisor.start()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{args.command}: {exc}", file=sys.stderr)
        return 1
    return 0
