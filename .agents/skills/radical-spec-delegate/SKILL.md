---
name: radical-spec-delegate
description: >-
  Delegate code edits, multi-file implementations, refactors, boilerplate
  generation, and test-fix loops to the local ollama-worker MCP server using a
  self-contained Radical Spec while keeping the cloud agent focused on
  architecture and diff review.
---

# Radical Spec Delegation Workflow

Act strictly as **Principal Architect & Diff Reviewer**. Preserve cloud model quota by offloading execution loops to the local `ollama-worker` MCP server.

## When to Execute Directly vs. Delegate

- **Handle in the cloud:** Pure Q&A, architecture/design decisions, code review, or tiny edits (<10 lines in a single file).
- **Delegate to `ollama-worker`:** New modules, feature implementations, multi-file refactors, boilerplate generation, unit-test authoring, and iterative test-fix loops.

## Delegation Procedure

1. **Verify Local Models (if needed):**
   Call `ollama-worker:list_local_ollama_models` to confirm installed models and choose a tool-capable tag matching the task and machine memory limits below.

2. **Gather Minimal Context Only:**
   Read only the minimal interface definitions, type signatures, or file headers needed to specify the task accurately. Do not stream entire implementation files into cloud context unless required for architectural decisions.

3. **Author a Self-Contained Radical Spec:**
   Include every section below so the local worker can execute autonomously without follow-up questions:
   - **Objective:** Clear statement of the desired behavior or fix.
   - **Exact File Scope & Do-Not-Touch List:** Explicit list of files to create/edit and files/directories that must remain untouched.
   - **Contracts & Invariants:** Exact function/class signatures, types, error-handling expectations, and edge cases.
   - **Ordered Step-by-Step Plan:** Sequential, concrete implementation steps.
   - **Exact Verification Command:** Exact command the local worker should run (e.g., `pytest tests/test_foo.py -q`) and expected pass condition.
   - **Concise Return Format:** Ask for a brief summary of files changed, commands run, and verification output.

4. **Invoke the Local Worker:**
   Call `ollama-worker:delegate_coding_task` with:
   - `spec`: The complete Radical Spec.
   - `workspace_path`: Absolute path to the target repository/workspace.
   - `model`: Chosen local Ollama model tag.

5. **Verify via Diff, Not Full File Reads:**
   Inspect `git status` and `git diff` (plus run the verification command if needed) rather than re-reading entire modified files in the cloud. If fixes are needed, send a concise, targeted follow-up Radical Spec back to `ollama-worker:delegate_coding_task`.

## If `ollama-worker` or Ollama Is Not Active

- **`ollama-worker` MCP tools are missing:** `ollama-worker` runs as a **STDIO** MCP server spawned automatically by Antigravity from `~/.gemini/config/mcp_config.json`. Do **not** launch `server.py` manually in a terminal or background shell (a separate process cannot attach to Antigravity's STDIO pipe). Instead, ask the user to click **Refresh** in **Installed MCP Servers** (or run `/mcp` in CLI).
- **Local Ollama daemon is unreachable (`http://localhost:11434`):** If `list_local_ollama_models` or `delegate_coding_task` reports Ollama is not running, start the Ollama daemon (`open -a Ollama` on macOS, or `ollama serve`) before retrying.

## Local Model Selection & Machine Memory (16 GB Reference)

- **Memory ceiling:** On 16 GB unified-memory / VRAM machines, keep model weights + KV cache under ~12–13 GB so execution stays `100% GPU`. Avoid models exceeding available memory (e.g., 23 GB+ models on 16 GB RAM) unless explicitly requested — they spill into CPU/swap offload.
- **Tool-calling required:** `delegate_coding_task` runs an agentic tool loop, so only select models with `tools` capability. Never pass embedding models (`nomic-embed-text`, `bge-m3`, `all-minilm`) or completion-only models (`gemma2:9b`).

| Model Tag | Size / Quant | Capabilities | Best For |
| :--- | :--- | :--- | :--- |
| `gemma4-32k:latest` *(default)* | `9.6 GB` (`8.0B Q4_K_M`) | `tools`, `thinking`, `num_ctx=32768` | Multi-file features, refactors, complex debugging, larger Radical Specs |
| `qwen2.5-coder:7b` | `4.7 GB` (`7.6B Q4_K_M`) | `tools` | Fast single-file edits, unit tests, iterative `pytest`/`ruff` loops, tight RAM |
| `llama3.1:8b-instruct-q8_0` | `8.5 GB` (`8.0B Q8_0`) | `tools` | Strict schema/formatting adherence and instruction-following |
| `gemma4:latest` | `9.6 GB` (`8.0B Q4_K_M`) | `tools`, `thinking` | Fallback only — prefer `gemma4-32k:latest` for explicit 32K `num_ctx` |
