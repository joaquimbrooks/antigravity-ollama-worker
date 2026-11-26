# antigravity-ollama-worker

A ToS-safe, zero-cloud-token local coding delegation worker for **Google Antigravity**, built on `mcp-harness` (`BaseProvider` + `MCPHarness`) and the official `google-antigravity` Python SDK (`LocalOpenAIAgentConfig.lightweight()`).

Antigravity (cloud planner) writes a self-contained **Radical Spec** and delegates implementation, boilerplate, unit tests, and test-fix loops to a local **Ollama** model over STDIO MCP, then verifies the result via `git diff`.

---

## Architecture

```text
Antigravity (Cloud Planner)
   │  STDIO JSON-RPC 2.0 (mcp-harness ProtocolEngine + StdioTransport)
   ▼
ollama-worker/server.py (MCPHarness + LoggingMiddleware + MetricsMiddleware)
   │
   ├── OllamaWorkerProvider(BaseProvider)   (ollama-worker/provider.py)
   │     ├── @tool list_local_ollama_models
   │     ├── @tool delegate_coding_task  ──► google-antigravity Agent(LocalOpenAIAgentConfig) ──► Ollama
   │     ├── @resource ollama://models
   │     └── @prompt radical_spec
   │
   └── policies.py (policy.allow_all() + policy.deny("run_command", ...))
```

## Quick Start

1. **Create a virtual environment & install dependencies:**
   ```zsh
   cd ollama-worker
   python3 -m venv .venv
   .venv/bin/python -m pip install --upgrade pip
   .venv/bin/python -m pip install google-antigravity pyyaml
   .venv/bin/python -m pip install --no-deps -e /path/to/mcp-harness
   ```

2. **Ensure Ollama is running with a tool-capable model:**
   ```zsh
   ollama pull gemma4-32k:latest   # or qwen2.5-coder:7b
   ollama list
   ```

3. **Run the verification suite:**
   ```zsh
   # Offline protocol, STDIO wire purity, and safety policy checks:
   ollama-worker/.venv/bin/python ollama-worker/test_worker.py --offline-only

   # Full smoke test including live Ollama agent execution:
   ollama-worker/.venv/bin/python ollama-worker/test_worker.py
   ```

4. **Register in Antigravity (Plugin or `mcp_config.json`):**
   - **Recommended (toggleable global plugin):** Install under `~/.gemini/config/plugins/ollama-worker/` with `plugin.json`, `mcp_config.json`, `skills/radical-spec-delegate/SKILL.md`, and `rules/AGENTS.md` so you can enable or disable the MCP server, skill, and rule together from Antigravity's **Customizations** UI.
   - **Standalone `mcp_config.json`:** Replace `/path/to/antigravity-ollama-worker` in `ollama-worker/mcp_config.snippet.json` with your repo path (from the repo root, you can preview the exact JSON with `sed "s|/path/to/antigravity-ollama-worker|$PWD|g" ollama-worker/mcp_config.snippet.json`).

See [CHECKLIST.md](CHECKLIST.md) for Ollama memory/context tuning (`OLLAMA_CONTEXT_LENGTH=32768`), verification steps, and troubleshooting.

---

*Co-authored with Google Antigravity.*
