#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
system="$(uname -s)"
machine="$(uname -m)"

case "$system" in
  Darwin)
    if [[ "$machine" != "arm64" ]]; then
      echo "Intel Macs are not supported." >&2
      exit 1
    fi
    if ! command -v ollama >/dev/null 2>&1; then
      if ! command -v brew >/dev/null 2>&1; then
        echo "Install Homebrew first: https://brew.sh" >&2
        exit 1
      fi
      brew install ollama
    fi
    ;;
  Linux)
    if ! command -v ollama >/dev/null 2>&1; then
      if ! command -v apt-cache >/dev/null 2>&1 ||
         ! apt-cache policy ollama | grep -Eq 'Candidate: [^ (]'; then
        echo "Ollama is unavailable in configured APT sources; see https://docs.ollama.com/linux" >&2
        exit 1
      fi
      sudo apt-get install -y ollama
    fi
    ;;
  *)
    echo "Unsupported operating system: $system" >&2
    exit 1
    ;;
esac

if ! command -v python3 >/dev/null 2>&1 ||
   ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
  echo "Python 3.11 or newer is required." >&2
  exit 1
fi
if [[ ! -e .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -c 'import sys; sys.exit(sys.version_info < (3, 11))' || {
  echo "Existing .venv requires Python 3.11 or newer." >&2
  exit 1
}
.venv/bin/python -m pip install -e .
if [[ ! -e .env && ! -L .env ]]; then
  (umask 077; cp -n .env.example .env)
  echo "Created .env from .env.example."
fi
.venv/bin/python -c '
import os
import secrets
import stat

fd = os.open(".env", os.O_RDWR | os.O_NOFOLLOW)
with os.fdopen(fd, "r+") as env:
    if not stat.S_ISREG(os.fstat(env.fileno()).st_mode):
        raise ValueError(".env must be a regular file")
    text = env.read()
    key = ""
    for line in text.splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name.strip() == "API_KEY":
            key = value.strip().strip("\"'\''")
    if not key.strip():
        os.fchmod(env.fileno(), 0o600)
        env.seek(0, os.SEEK_END)
        env.write(("" if not text or text.endswith("\n") else "\n") + "API_KEY=" + secrets.token_hex(32) + "\n")
        print("Generated API_KEY in .env.")
'
echo "Initialization complete. For fish: source .venv/bin/activate.fish"
echo "For bash/zsh: source .venv/bin/activate"
