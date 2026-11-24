# Ollama Model Names Implementation Plan

**Goal:** Use model-derived Ollama IDs without weakening validation or breaking gateway recovery.

**Architecture:** Retain the pinned artifact and backend checks; extend local manifest lookup to safe name:tag IDs. Persist only validated enabled IDs, and migrate the existing model by copying it, probing it, and activating it before deleting its old name.

**Tech Stack:** Python, Ollama API, unittest.

**Spec:** Approved chat description: model-name:quantization IDs, private cached identity, fresh validation, preserve old model on migration failure.

## Tasks

- [ ] Add failing tests for safe tagged manifest lookup, tampered blobs, tagged probe identity and persisted enabled IDs.
- [ ] Implement safe tagged IDs throughout identity, probe response checking, CLI, and supervisor persistence without trusting cache as evidence.
- [ ] Document the naming and verification limitations; run the complete test suite.
- [ ] Copy the existing model to `deepseek-v4-pro-qwen3.5-9b-uncensored:q4_k_m`, restart the supervisor, probe and enable it, test gateway and restart restoration.
- [ ] Only after successful checks remove the old Ollama ID and update the OpenCode model display configuration; verify discovery.
