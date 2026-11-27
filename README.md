# antigravity-ollama-worker

A ToS-safe, zero-cloud-token local coding delegation worker for **Google Antigravity**, backed by the official `google-antigravity` Python SDK (`LocalOpenAIAgentConfig.lightweight()`) with optional plug-in support for a custom `mcp-harness` runtime.

Antigravity (cloud planner) writes a self-contained **Radical Spec** and delegates implementation, boilerplate, unit tests, and test-fix loops to a local **Ollama** model over STDIO MCP, then verifies the result via `git diff`.

---

## Architecture

```text
Antigravity (Cloud Planner)
   │  STDIO JSON-RPC 2.0
   ▼
ollama-worker/server.py
   ├── If `mcp_harness` is installed:
   │     MCPHarness + LoggingMiddleware + MetricsMiddleware + OllamaWorkerProvider(BaseProvider)
   └── Default fallback (zero extra deps):
         mcp.server.mcpserver.MCPServer (bundled with google-antigravity)
   │
   ├── @tool list_local_ollama_models
   ├── @tool get_worker_scaling_profile
   ├── @tool delegate_coding_task  ──► WorkerSlotGate ──► Agent(LocalOpenAIAgentConfig) ──► Ollama
   ├── @resource ollama://models (includes host_profile)
   ├── @prompt radical_spec
   ├── hardware.py (host profiling, OLLAMA_WORKER_* overrides, warnings & fcntl slot gate)
   └── policies.py (policy.allow_all() + policy.deny("run_command", ...))
```

### Optional Bring-Your-Own Harness (`mcp-harness`)

`google-antigravity>=0.1.24` is the **only required dependency**—`server.py` falls back automatically to the standard `MCPServer` bundled with `google-antigravity`.

If you use a custom MCP runtime implementing the `mcp-harness` interface (`mcp_harness.providers.base.BaseProvider` with `@tool`/`@resource`/`@prompt` decorators, `mcp_harness.server.MCPHarness`, middleware hooks `use_logging()` / `use_metrics()`, and YAML config loading via `ollama-worker/harness.yml`), `server.py` automatically upgrades to `MCPHarness` when `mcp_harness` is importable in the virtual environment.

---

## Quick Start

1. **Create a virtual environment & install dependencies:**
   ```zsh
   cd ollama-worker
   python3 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install google-antigravity
   ```
   *(Optional: if you have a local `mcp-harness` package, also install `pyyaml` and `pip install --no-deps -e /path/to/your-mcp-harness`.)*

2. **Ensure Ollama is running with a tool-capable model:**
   ```zsh
   ollama pull gemma4-32k:latest   # or qwen2.5-coder:7b
   ollama list
   ```

3. **Run the verification suite:**
   ```zsh
   # Offline protocol, fallback, STDIO wire purity, and safety policy checks:
   ollama-worker/.venv/bin/python ollama-worker/test_worker.py --offline-only

   # Full smoke test including live Ollama agent execution:
   ollama-worker/.venv/bin/python ollama-worker/test_worker.py
   ```

4. **Install into Antigravity (Recommended: Toggleable Plugin, Single Source of Truth):**
   Run the idempotent installer from the repo root to package `plugin/`, `.agents/`, and a host-tuned `mcp_config.json` into `~/.gemini/config/plugins/ollama-worker/`:
   ```zsh
   ./install_plugin.sh
   ```
   To customize parallelism/cores/memory or apply recommended Ollama daemon settings (`launchctl setenv` on macOS):
   ```zsh
   ./install_plugin.sh --max-cores 2 --max-parallel 2 --tune-ollama
   ```
   Re-run `./install_plugin.sh` after editing `.agents/skills/` or `.agents/rules/` to keep your global plugin synced without config drift. You can then enable or disable the entire bundle (MCP server, skill, and rule) from Antigravity's **Customizations** UI.

   *(Alternative standalone config: preview resolved JSON with `sed "s|/path/to/antigravity-ollama-worker|$PWD|g" ollama-worker/mcp_config.snippet.json` and merge it into `~/.gemini/config/mcp_config.json`.)*

### Dynamic Host Scaling, Overrides & Warnings

`ollama-worker/hardware.py` inspects the host OS, CPU performance cores, and unified memory / GPU VRAM to select safe defaults (`compact-<=16gb`, `mid-24gb-36gb`, `high-48gb+`), enforced across STDIO subagents by `WorkerSlotGate` (`fcntl.flock` slot locks) and inspectable via `get_worker_scaling_profile` or `python3 ollama-worker/hardware.py --json`:

- **Customization flags (`./install_plugin.sh`) & env vars (`OLLAMA_WORKER_*`):**
  - `--max-parallel <N>` / `OLLAMA_WORKER_MAX_PARALLEL` — override max concurrent worker slots (`>= 1`)
  - `--max-cores <N>` / `OLLAMA_WORKER_MAX_CORES` — override detected performance cores and cap auto parallelism
  - `--memory-gb <GB>` / `OLLAMA_WORKER_MEMORY_GB` — override detected RAM/VRAM
  - `--context-length <N>` / `OLLAMA_WORKER_CONTEXT_LENGTH` — override `OLLAMA_CONTEXT_LENGTH` (`>= 4096`)
  - `--model <tag>` / `OLLAMA_WORKER_MODEL` — default single-worker model
  - `--parallel-model <tag>` / `OLLAMA_WORKER_PARALLEL_MODEL` — recommended model for concurrent subagents
  - `--tune-ollama` — apply recommended `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_KV_CACHE_TYPE`, and `OLLAMA_FLASH_ATTENTION` (`launchctl` on macOS; prints exports elsewhere)
- **Built-in Warnings:**
  - **KV-cache multiplication:** On `<= 16 GB` hosts (which default to `max_parallel_workers=1` with `gemma4-32k:latest`), opting into `OLLAMA_NUM_PARALLEL=2` allocates `2x` KV cache — use a `<= 5 GB` native tool-calling model (`qwen3:4b` or `gemma4:e4b`) and keep `OLLAMA_MAX_LOADED_MODELS=1` to avoid swap spill or model thrashing.
  - **Raw-JSON tool-call guard:** `qwen2.5-coder:7b` frequently emits raw JSON text instead of structured `tool_calls` under `LocalOpenAIAgentConfig`; `provider.py` detects unexecuted JSON tool payloads and returns `status=tool_call_error` instead of silent false-success.
  - **Workspace collisions:** Running two concurrent `delegate_coding_task` calls in the same directory emits a collision warning; assign disjoint files or separate `git worktree` paths.

### Using with Other MCP Clients (Codex CLI, Claude Code, Cursor, etc.)

Because `ollama-worker/server.py` speaks standard **STDIO JSON-RPC 2.0 MCP**, any MCP-capable client can mount it with the same `command` + `args` from `ollama-worker/mcp_config.snippet.json` (and `.agents/skills/radical-spec-delegate/SKILL.md` follows the portable `SKILL.md` format).

When calling `delegate_coding_task` from any client or harness:
- Pass an **absolute** `workspace_path` and a self-contained **Radical Spec** (objective, file scope, invariants, ordered steps, and verification command).
- `policies.py` enforces the `run_command` safety guard locally regardless of which MCP client initiated the request.

See [CHECKLIST.md](CHECKLIST.md) for Ollama memory/context tuning (`OLLAMA_CONTEXT_LENGTH=32768`), verification steps, and troubleshooting.

---

*Co-authored with Google Antigravity.*
