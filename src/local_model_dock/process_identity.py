"""Fail-closed evidence that a supervised child owns a loopback listener."""

import ipaddress
import os
import struct
import subprocess
import sys
from pathlib import Path


def process_owns_loopback_port(process, port, host="127.0.0.1"):
    """Check the live Popen child owns every listener on port, including host."""
    try:
        address = ipaddress.ip_address(host)
        if not address.is_loopback or type(port) is not int or not 1 <= port <= 65535:
            return False
        pid = process.pid
        if type(pid) is not int or pid <= 0 or process.poll() is not None:
            return False
        if sys.platform == "darwin":
            verified = _lsof_owns(pid, port, address)
        elif sys.platform.startswith("linux"):
            verified = _proc_owns(pid, port, address)
        else:
            return False
        return verified and process.poll() is None
    except (OSError, ValueError, TypeError, AttributeError, subprocess.SubprocessError):
        return False


def _lsof_owns(pid, port, address):
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpn"],
        capture_output=True, text=True, timeout=5, check=False,
    )
    if result.returncode != 0:
        return False
    owner = None
    found = False
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            owner = int(line[1:])
            if owner != pid:
                return False
        elif line.startswith("n"):
            if owner != pid:
                return False
            name = line[1:]
            expected = f"[{address}]:{port}" if address.version == 6 else f"{address}:{port}"
            found |= name == expected
        elif line.startswith("f") and owner == pid and line[1:].isdigit():
            continue
        else:
            return False
    return found


def _proc_owns(pid, port, address):
    inodes = set()
    found = False
    for family in ("tcp", "tcp6"):
        lines = Path(f"/proc/net/{family}").read_text().splitlines()
        if not lines:
            return False
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 10:
                return False
            local, state, inode = fields[1], fields[3], fields[9]
            raw_address, raw_port = local.split(":")
            if int(raw_port, 16) != port or state != "0A":
                continue
            size = 8 if family == "tcp" else 32
            if len(raw_address) != size or not inode.isdecimal():
                return False
            packed = b"".join(
                struct.pack("=I", int(raw_address[i:i + 8], 16))
                for i in range(0, size, 8)
            )
            local_address = ipaddress.ip_address(packed)
            found |= local_address == address
            inodes.add(inode)
    if not found:
        return False
    descriptors = Path(f"/proc/{pid}/fd")
    owned = set()
    for fd in descriptors.iterdir():
        target = os.readlink(fd)
        if target.startswith("socket:[") and target.endswith("]"):
            owned.add(target[8:-1])
    return bool(inodes) and inodes <= owned
