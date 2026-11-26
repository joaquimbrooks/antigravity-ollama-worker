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
   ├── @tool delegate_coding_task  ──► google-antigravity Agent(LocalOpenAIAgentConfig) ──► Ollama
   ├── @resource ollama://models
   ├── @prompt radical_spec
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
   Run the idempotent installer from the repo root to package `plugin/`, `.agents/`, and a resolved `mcp_config.json` into `~/.gemini/config/plugins/ollama-worker/`:
   ```zsh
   ./install_plugin.sh
   ```
   Re-run `./install_plugin.sh` after editing `.agents/skills/` or `.agents/rules/` to keep your global plugin synced without config drift. You can then enable or disable the entire bundle (MCP server, skill, and rule) from Antigravity's **Customizations** UI.

   *(Alternative standalone config: preview resolved JSON with `sed "s|/path/to/antigravity-ollama-worker|$PWD|g" ollama-worker/mcp_config.snippet.json` and merge it into `~/.gemini/config/mcp_config.json`.)*

### Using with Other MCP Clients (Codex CLI, Claude Code, Cursor, etc.)

Because `ollama-worker/server.py` speaks standard **STDIO JSON-RPC 2.0 MCP**, any MCP-capable client can mount it with the same `command` + `args` from `ollama-worker/mcp_config.snippet.json` (and `.agents/skills/radical-spec-delegate/SKILL.md` follows the portable `SKILL.md` format).

When calling `delegate_coding_task` from any client or harness:
- Pass an **absolute** `workspace_path` and a self-contained **Radical Spec** (objective, file scope, invariants, ordered steps, and verification command).
- `policies.py` enforces the `run_command` safety guard locally regardless of which MCP client initiated the request.

See [CHECKLIST.md](CHECKLIST.md) for Ollama memory/context tuning (`OLLAMA_CONTEXT_LENGTH=32768`), verification steps, and troubleshooting.

---

*Co-authored with Google Antigravity.*
