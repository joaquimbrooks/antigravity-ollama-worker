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
   - **Short micro-specs (`<= 4` turns / fast models like `qwen3:8b`):** Call `ollama-worker:delegate_coding_task` with `spec`, `workspace_path`, and `model` (optionally `disable_thinking=True`, `idle_timeout_seconds`, `max_wall_seconds`).
   - **Longer specs or slower reasoning models (e.g., `gemma4-32k:latest` on base M1):** Use the timeout-proof async job workflow so MCP's 3-minute call ceiling never cancels an active run:
     1. Call `ollama-worker:start_coding_task` (`spec`, `workspace_path`, `model`, plus optional `disable_thinking`, `idle_timeout_seconds`, `max_wall_seconds`) to receive a `job_id` immediately.
     2. Poll `ollama-worker:check_coding_task(job_id=..., wait_seconds=30)` to inspect live progress (`phase`, `idle`, `chunks`, `thinking_chunks`, `tool_calls`, `turns`, `last_tool`) until `status=completed` (or `tool_call_error` / `timeout` / `error`).
     3. Call `ollama-worker:cancel_coding_task(job_id=...)` to abort early (force-stops the underlying harness subprocess and confirms `status=cancelled` once the worker slot is released, or `status=cancelling` while still unwinding).

5. **Verify via Diff, Not Full File Reads:**
   Inspect `git status` and `git diff` (plus run the verification command if needed) rather than re-reading entire modified files in the cloud. If fixes are needed, send a concise, targeted follow-up Radical Spec back to `ollama-worker`.

## If `ollama-worker` or Ollama Is Not Active

- **`ollama-worker` MCP tools are missing:** `ollama-worker` runs as a **STDIO** MCP server spawned automatically by Antigravity from `~/.gemini/config/mcp_config.json`. Do **not** launch `server.py` manually in a terminal or background shell (a separate process cannot attach to Antigravity's STDIO pipe). Instead, ask the user to click **Refresh** in **Installed MCP Servers** (or run `/mcp` in CLI).
- **Local Ollama daemon is unreachable (`http://localhost:11434`):** If `list_local_ollama_models` or `delegate_coding_task` reports Ollama is not running, start the Ollama daemon (`open -a Ollama` on macOS, or `ollama serve`) before retrying.

## Dynamic Host Scaling & Parallel Subagents

- **Inspect host capacity first:** Call `ollama-worker:list_local_ollama_models` or `ollama-worker:get_worker_scaling_profile` to view the detected host tier, effective cores/memory, `max_parallel_workers`, recommended single vs. parallel models, and any active warnings.
- **Keep Radical Specs compact (`<= 4` tool turns / 1–3 files):** Synchronous `delegate_coding_task` runs over MCP with a ~3-minute call timeout. Use `start_coding_task` + `check_coding_task` for larger specs or slower reasoning models like `gemma4-32k:latest`.
- **16 GB hosts default to single-worker (`max_parallel_workers=1`, `OLLAMA_CONTEXT_LENGTH=16384`):** `gemma4-32k:latest` (`~9.0 GB`) and `qwen3:8b` (`~5.2 GB`) support reliable native OpenAI `tool_calls`, while `16384` context keeps KV-cache pressure low on 16 GB unified memory (opt into `32768` with `--context-length 32768` only when large context is needed). Avoid `qwen2.5-coder:7b` and `llama3.1:8b-instruct-q8_0` for multi-turn agentic loops if they emit plain-text JSON blocks instead of `tool_calls` (flagged as `status=tool_call_error`).
- **Opting into 2x parallel on `<= 16 GB`:** Because `OLLAMA_NUM_PARALLEL=2` allocates `2x` KV cache, only enable `--max-parallel 2` with a `<= 5 GB` native tool-calling model (e.g., `qwen3:4b` or `gemma4:e4b`) and keep all concurrent workers on the **same** model tag (`OLLAMA_MAX_LOADED_MODELS=1`) to prevent weight thrashing or swap spill.
- **Isolate parallel edits:** Never run two parallel coding tasks against overlapping files in the same workspace. Assign strictly disjoint file scopes or separate `git worktree` paths (the worker slot gate emits a collision warning when concurrent tasks share a workspace).
- **Host overrides:** Tune or cap host scaling without editing code via `OLLAMA_WORKER_MAX_PARALLEL`, `OLLAMA_WORKER_MAX_CORES`, `OLLAMA_WORKER_MEMORY_GB`, `OLLAMA_WORKER_CONTEXT_LENGTH`, `OLLAMA_WORKER_MODEL`, and `OLLAMA_WORKER_PARALLEL_MODEL` (or `./install_plugin.sh --max-cores <N> --max-parallel <N> --tune-ollama`).
- **Tool-calling required:** Agentic delegation requires reliable structured `tool_calls` support. Never pass embedding models (`nomic-embed-text`, `bge-m3`, `all-minilm`) or completion-only models (`gemma2:9b`).

| Model Tag | Size / Quant | Capabilities | Best For |
| :--- | :--- | :--- | :--- |
| `gemma4-32k:latest` *(default)* | `8.95 GB` (`8.0B Q4_K_M`) | `tools`, `thinking`, `num_ctx=32768` | Complex multi-file reasoning (prefer `start_coding_task` + `check_coding_task` on base M1) |
| `qwen3:8b` *(fast M1 tool-caller)* | `~5.2 GB` (`8.2B Q4_K_M`) | `tools`, `thinking` | Fast native tool-calling micro-specs on 16 GB machines well under 3m0s |
| `qwen3:4b` / `gemma4:e4b` *(<=16 GB parallel)* | `~2.5–4.5 GB` | `tools`, `thinking` | Small native tool-calling models if opting into `--max-parallel 2` on <=16 GB hosts |
| `llama3.1:8b-instruct-q8_0` | `7.95 GB` (`8.0B Q8_0`) | `tools` | Strict schema/formatting adherence and instruction-following |
| `qwen2.5-coder:14b` *(24–48 GB+ hosts)* | `~9–10 GB` (`14B Q4_K_M`) | `tools` | High-memory hosts (48 GB+ or dedicated 24 GB+ VRAM) |
| `qwen2.5-coder:7b` *(avoid for agent loops)* | `4.36 GB` (`7.6B Q4_K_M`) | `tools` *(flaky JSON)* | Single-turn completion only — frequently emits raw JSON text instead of `tool_calls` |
| `gemma4:latest` | `8.95 GB` (`8.0B Q4_K_M`) | `tools`, `thinking` | Fallback only — prefer `gemma4-32k:latest` for explicit 32K `num_ctx` |
