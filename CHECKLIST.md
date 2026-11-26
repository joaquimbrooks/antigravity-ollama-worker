# Ollama Worker Setup & Verification Checklist

## Prerequisites & Environment

- **OS:** macOS, Linux, or Windows
- **Python:** `>=3.10`
- **Virtual Environment:** `<REPO_ROOT>/ollama-worker/.venv` (`google-antigravity`, `mcp-harness`, `pyyaml`)
- **Ollama:** Installed and reachable at `http://localhost:11434/api/tags` with at least one tool-capable local model

---

## 1. Ollama Tuning (Memory & Context)

On **16 GB** unified-memory / VRAM machines (< 24 GB), configure a `32768` context window (`OLLAMA_CONTEXT_LENGTH=32768`) with quantized KV cache and flash attention so tool-heavy coding loops stay `100% GPU` without spilling to swap. (Machines with 24 GB+ VRAM can increase `OLLAMA_CONTEXT_LENGTH` or run larger models.)

- [ ] Set the Ollama environment variables (macOS GUI app via `launchctl`, or equivalent service env on Linux/Windows):
  ```zsh
  launchctl setenv OLLAMA_CONTEXT_LENGTH 32768
  launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
  launchctl setenv OLLAMA_FLASH_ATTENTION 1
  ```
- [ ] *(Optional, if launching `ollama serve` manually from Terminal)* Add these exports to your shell profile (`~/.zshrc` or `~/.bashrc`):
  ```zsh
  export OLLAMA_CONTEXT_LENGTH=32768
  export OLLAMA_KV_CACHE_TYPE=q8_0
  export OLLAMA_FLASH_ATTENTION=1
  ```
- [ ] Fully quit and restart Ollama so the new environment variables take effect:
  ```zsh
  pkill -x Ollama || true
  pkill -x ollama || true
  open -a Ollama
  ```
- [ ] Load a coding model briefly and verify `CONTEXT` (`32768`) and `100% GPU` via `ollama ps`:
  ```zsh
  ollama run gemma4-32k:latest "Reply with OK"
  ollama ps
  ```

---

## 2. Tool-Capable Model Selection

`ollama-worker/provider.py` defaults to `gemma4-32k:latest` (override via `OLLAMA_WORKER_MODEL`). Always choose a model that supports Ollama `tools` calling and fits within your available VRAM/RAM:

- [ ] Pull one or more tool-capable coding models sized for your machine:
  ```zsh
  ollama pull qwen2.5-coder:7b      # ~4.7 GB — fast for single-file edits & test loops
  ollama pull gemma4-32k:latest     # ~9.6 GB — strong default for 16 GB machines
  # Optional on 16 GB+ machines:
  ollama pull qwen2.5-coder:14b
  ```
- [ ] Avoid delegating to models larger than available VRAM (which causes slow CPU/GPU split offloading) or non-tool / embedding models (e.g., `gemma2:9b`, `nomic-embed-text`, `bge-m3`).

---

## 3. Registering `ollama-worker` in Antigravity

- [ ] Generate your parameterized config snippet from the repository root:
  ```zsh
  sed "s|/path/to/antigravity-ollama-worker|$PWD|g" ollama-worker/mcp_config.snippet.json
  ```
- [ ] Merge the `"ollama-worker"` entry from [mcp_config.snippet.json](ollama-worker/mcp_config.snippet.json) (with `/path/to/antigravity-ollama-worker` replaced by your local repo path) into your Antigravity MCP config (`~/.gemini/config/mcp_config.json` on macOS/Linux, `%USERPROFILE%\.gemini\config\mcp_config.json` on Windows, or `.agents/mcp_config.json` for a single workspace):
  ```json
  {
    "mcpServers": {
      "ollama-worker": {
        "command": "/path/to/antigravity-ollama-worker/ollama-worker/.venv/bin/python",
        "args": [
          "/path/to/antigravity-ollama-worker/ollama-worker/server.py"
        ]
      }
    }
  }
  ```
- [ ] *(Optional — `mcp-harness` CLI / YAML mode)* Launch `OllamaWorkerProvider` via `mcp-harness` directly (or compose it with other `mcp-harness` providers in [harness.yml](ollama-worker/harness.yml)):
  ```zsh
  PYTHONPATH="$PWD/ollama-worker" \
  "$PWD/ollama-worker/.venv/bin/python" \
  -m mcp_harness -c "$PWD/ollama-worker/harness.yml"
  ```
- [ ] In Antigravity IDE/App, open **Additional Options (...) > MCP Servers** (or **Installed MCP Servers**) and click **Refresh** to connect `ollama-worker`. If using the `agy` CLI, run `/mcp` to verify `ollama-worker` is connected and exposes `list_local_ollama_models`, `delegate_coding_task`, resource `ollama://models`, and prompt `radical_spec`.

---

## 4. Global Plugin Install (One-Click Toggle)

The Skill and Rule live in `.agents/` as source templates. To enable them across all repositories with a single UI/CLI on-off toggle, package them as an Antigravity plugin under `~/.gemini/config/plugins/ollama-worker/`:

- `plugin.json` — plugin manifest (`ollama-worker`)
- `mcp_config.json` — STDIO `ollama-worker` server config (with your local repo path; move `"ollama-worker"` out of `~/.gemini/config/mcp_config.json` so toggling the plugin controls everything together)
- `skills/radical-spec-delegate/SKILL.md` — copied from `.agents/skills/radical-spec-delegate/SKILL.md`
- `rules/AGENTS.md` — copied from `.agents/rules/quota-delegation.md`

---

## 5. Verification Tests

- [ ] **Offline Protocol & Policy Check:**
  ```zsh
  ./ollama-worker/.venv/bin/python ./ollama-worker/test_worker.py --offline-only
  ```
- [ ] **Full Smoke Test (`test_worker.py`):**
  Run the standalone smoke test against your local Ollama instance:
  ```zsh
  ./ollama-worker/.venv/bin/python ./ollama-worker/test_worker.py
  ```
- [ ] **MCP Tool Discovery in Antigravity:**
  In a new Antigravity chat, ask:
  > "Call `list_local_ollama_models` from `ollama-worker` and show me the installed models."
- [ ] **Real Delegated Coding Task:**
  Ask Antigravity to implement a small multi-file feature or unit test in a git-tracked workspace using `radical-spec-delegate`, then verify:
  1. `ollama ps` shows the active coding model (`100% GPU` and `32768` context).
  2. `git diff` in the target workspace shows the expected clean local edits.

---

## 6. Troubleshooting Quick Checks

- **Stdio JSON framing errors / MCP disconnect on startup:**
  Ensure nothing prints to `stdout` in `server.py`. Confirm `logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)` stays at the very top of `ollama-worker/server.py` before any `google.antigravity` or `mcp_harness` imports.
- **Model forgets instructions or truncates tool calls:**
  Run `ollama ps` during a task and check `CONTEXT`. If it shows `4096` or `8192` instead of `32768`, `launchctl setenv` was not picked up—fully quit Ollama (`pkill -x Ollama`) and relaunch it.
- **Slow generation / CPU-GPU offload split (`CPU/GPU` split in `ollama ps`):**
  The loaded model + `32768` KV cache exceeds available VRAM. Switch to a smaller model (e.g., `qwen2.5-coder:7b` or `gemma4-32k:latest`), or lower `OLLAMA_CONTEXT_LENGTH` to `16384`.
- **Stale `.venv` or wrong Python path:**
  Verify `./ollama-worker/.venv/bin/python --version` works and matches the `"command"` path in your `mcp_config.json`.
