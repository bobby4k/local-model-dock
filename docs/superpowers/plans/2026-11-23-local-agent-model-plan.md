# Local Model Dock Implementation Plan

> **For agentic workers:** Use `executing-plans` to implement the tasks in order. Checkboxes track progress; never claim unimplemented features work.

**Goal:** Safely discover, validate, run and update hardware-compatible MoE models for local-network Agent use.

**Architecture:** A Python CLI manages isolated backend processes, persistent candidate/active/rollback state, and a guarded OpenAI-compatible gateway. Discovery and decision logic are separate from backend-specific operations, with conservative handling of unknowns and confirmation-bound updates.

**Tech Stack:** Python 3.11+, oMLX on Apple Silicon, Ollama on Linux, Hugging Face Hub and Ollama model sources.

**Spec:** `docs/superpowers/specs/2026-11-23-agent-model-dock-design.md`.

## Global Constraints

- Apple Silicon and Linux only; no Intel Mac. `.env` budget defaults to 20 GB as a warning threshold.
- MoE only; formal recommendations require trusted tool-call validation bound to exact model and backend versions.
- DeepSeek and Qwen priority configurable; other families are fallback. Unknown resource or architecture data is not treated as adequate.
- `start`, `update`, `stop`, `restart` required. First-download auto mode defaults on but stops for uncertain resources or anticipated budget excess.
- Updates require notification and specific confirmation before download; current and most recent previous model remain available through download and verification.
- Telegram token and authorized user/chat IDs are optional `.env` settings. Local confirmation works without Telegram.
- Network-facing gateway requires authentication; backend ports must not bypass it. Do not expose secrets in logs, Git, or examples.
- Do not delete files outside project management; do not install system dependencies or modify firewall or cron.

## Review Focus

- Unauthorized or malformed gateway requests cannot access backend or disclose credentials (Task 5).
- Interrupted or repeated updates must retain the active model and reject stale approval (Task 6).
- Unreliable size or hardware metadata cannot trigger unattended download (Tasks 2–3).
- Running streams must not be interrupted by a model switch when backend permits draining (Tasks 5–6).
- Shared backend files and budget accounting cannot cause deletion of unmanaged data (Tasks 3–4).

## File Structure

- `pyproject.toml`, `src/local_model_dock/__main__.py`, `cli.py`, `config.py`: package, commands and configuration.
- `src/local_model_dock/hardware.py`, `discovery.py`, `selection.py`: platform detection, online metadata and policy.
- `src/local_model_dock/backend.py`, `storage.py`: isolated backend lifecycle, versioned model files and storage accounting.
- `src/local_model_dock/gateway.py`: authenticated API forwarding and stable model name.
- `src/local_model_dock/updates.py`, `telegram.py`: persistent approvals, update and optional notifications.
- `tests/`: tests for each module and integration flows.
- `README.md`, `.env.example`, `.gitignore`: operating instructions and safe configuration.

### Task 1: Package, configuration and command contract

**Files:** create package, `pyproject.toml`, `.gitignore`, `.env.example`, and `tests/test_cli.py`, `tests/test_config.py`.

- [ ] Write failing tests that CLI recognizes `start`, `update`, `stop`, `restart`, refuses unsupported operations safely, and parses `.env` defaults (20 GB, automatic first download enabled, Telegram disabled without token).
- [ ] Run `python3 -m unittest discover -s tests -v`; verify tests fail for missing package.
- [ ] Implement config parsing without printing secrets, platform-neutral CLI entry point, and safe explicit errors for lifecycle operations not yet implemented. Add `.env` and runtime data to `.gitignore`.
- [ ] Run the test suite and verify all Task 1 tests pass; confirm no secrets are present in tracked files.

### Task 2: Hardware and selection policy

**Files:** `hardware.py`, `selection.py`, tests for both.

- [ ] Write failing tests for Apple Silicon, Intel Mac rejection, Linux unknown GPU, memory unknown, MoE false/unknown, insufficient disk, verified tool calls, family fallback, priority ordering and warning threshold.
- [ ] Implement detection and deterministic selection, separating verified recommendations from untested discoveries; only automatic-download verified candidates with known adequate resources within budget.
- [ ] Run tests and verify unsupported and unknown cases fail closed.

### Task 3: Online discovery and version metadata

**Files:** `discovery.py`, `tests/test_discovery.py`.

- [ ] Mock paginated Hub responses, failed requests, missing architecture, quantization/file sizes, licenses and immutable revisions; assert unknowns never become verified facts.
- [ ] Implement read-only discovery of new MoE candidates and version identifiers; use source-specific APIs and label metadata of uncertain provenance. Verify Ollama registry discovery mechanism before relying on it.
- [ ] Run tests; document upstream limitations and do not claim discovery is complete if registry metadata is insufficient.

### Task 4: Isolated backends and storage

**Files:** `backend.py`, `storage.py`, corresponding tests.

- [ ] Test isolated model paths, missing dependencies, process ownership, safe start/stop/restart, caching configuration, model size plus temporary files, over-budget notifications and no deletion of external files.
- [ ] Implement Apple Silicon oMLX and Linux Ollama adapters; keep each backend loopback-only, use project-managed paths, configure capped oMLX SSD cache, and retain immutable model revisions.
- [ ] Verify backend lifecycle using mocked processes; test real backend operations only where dependencies and hardware are available.

### Task 5: Authenticated stable gateway

**Files:** `gateway.py`, `tests/test_gateway.py`.

- [ ] Test missing/wrong/correct API key, non-loopback backend bypass prevention, normal and streamed completions, tool-call forwarding, stable model mapping, and active-stream draining.
- [ ] Implement an authenticated LAN gateway; keep backend private, stream transparently and never log authorization or content.
- [ ] Run tests and a local end-to-end check. Document that HTTP is unencrypted and LAN exposure depends on firewall/network configuration.

### Task 6: Approval, update and rollback

**Files:** `updates.py`, `telegram.py`, tests for both.

- [ ] Test pending candidate persistence, notification failure, local confirmation without Telegram, unauthorized Telegram users/chats, expiration, replay, duplicate/concurrent updates, inadequate disk, download/test failure, restart recovery and rollback preservation.
- [ ] Implement candidate-bound one-use approvals, optional Telegram with allowlisted IDs, and a crash-safe update state machine. `update` checks and reports before any download; explicit approval triggers download and validation.
- [ ] Run tests with fake sources and backends; verify old active service remains available through failed operations.

### Task 7: Documentation and end-to-end verification

**Files:** `README.md`, `.env.example`, integration tests.

- [ ] Document commands, service prerequisites, cron example, `.env` settings, security, verified model limitations, manual confirmation, rollback and uninstall.
- [ ] Run full tests, check source for secrets, verify local gateway and CLI; perform real Apple Silicon and Linux smoke tests only on available systems.
- [ ] Record untested combinations and actual results; commit and push only after verification and review of the diff.
