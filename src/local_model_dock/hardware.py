import platform
import os
import re
import shutil
import subprocess
from pathlib import Path


def cpu_name() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=2
            ).strip() or "unknown"
        except (OSError, subprocess.SubprocessError):
            return "unknown"
    return platform.processor() or "unknown"


def total_memory() -> int | None:
    try:
        value = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return value if value > 0 else None
    except (OSError, ValueError):
        return None


def print_hardware() -> None:
    def size(value):
        return f"{value / 1024**3:.1f} GiB" if value is not None else "unknown"

    system, machine = platform.system(), platform.machine()
    try:
        backend = detect_backend(system, machine)
    except ValueError:
        backend = "unsupported"
    disk = available_disk(".")
    print(f"Hardware: {system} {machine} | CPU: {cpu_name()} | backend: {backend}")
    label = "immediately free" if system == "Darwin" else "available"
    print(f"Memory: {size(total_memory())} total, {size(available_memory())} {label}"
          f" | disk free: {size(disk)}")


def detect_backend(system: str | None = None, machine: str | None = None) -> str:
    system = system or platform.system()
    machine = machine or platform.machine()
    if system == "Darwin" and machine == "arm64":
        return "ollama"
    if system == "Linux":
        return "ollama"
    raise ValueError(f"Unsupported platform: {system} {machine}")


def available_memory(system: str | None = None) -> int | None:
    system = system or platform.system()
    if system == "Darwin":
        try:
            output = subprocess.check_output(["vm_stat"], text=True, timeout=2)
            page_sizes = re.findall(r"^Mach Virtual Memory Statistics: \(page size of ([0-9]+) bytes\)$", output, re.MULTILINE)
            free_pages = re.findall(r"^Pages free:\s*([0-9]+)\.$", output, re.MULTILINE)
            if len(page_sizes) == len(free_pages) == 1 and int(page_sizes[0]) > 0:
                # Free pages only: cached and compressed pages are not guaranteed reclaimable.
                free = int(page_sizes[0]) * int(free_pages[0])
                total = total_memory()
                if total is not None and free <= total:
                    return free
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return None
    if system != "Linux":
        return None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) == 3 and parts[2] == "kB":
                    return int(parts[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def available_disk(path: str | Path) -> int | None:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None
