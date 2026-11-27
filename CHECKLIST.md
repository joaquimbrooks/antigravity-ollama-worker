# Ollama Worker Setup & Verification Checklist

## Prerequisites & Environment

- **OS:** macOS, Linux, or Windows
- **Python:** `>=3.10`
- **Virtual Environment:** `<REPO_ROOT>/ollama-worker/.venv` (`google-antigravity`, plus optional `mcp-harness` + `pyyaml`)
- **Ollama:** Installed and reachable at `http://localhost:11434/api/tags` with at least one tool-capable local model

---

## 1. Dynamic Host Scaling, Overrides & Ollama Daemon Tuning

`ollama-worker/hardware.py` profiles the host OS, CPU performance cores, and unified memory / GPU VRAM at startup and computes a `WorkerScalingProfile` (`compact-<=16gb`, `mid-24gb-36gb`, or `high-48gb+`), enforcing concurrency across STDIO subagent processes via `WorkerSlotGate` (`fcntl.flock` slot locks).

| Host Tier | `OLLAMA_NUM_PARALLEL` | `OLLAMA_MAX_LOADED_MODELS` | `OLLAMA_CONTEXT_LENGTH` | Single Worker Model | Parallel Subagents Model |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `compact-<=16gb` *(e.g. 16 GB M1)* | `1` *(safe default)* | `1` | `32768` | `gemma4-32k:latest` | `qwen3:4b` *(opt-in via `--max-parallel 2`)* |
| `mid-24gb-36gb` | `2` *(if >=2 cores)* | `1` (<30 GB) / `2` | `32768` | `gemma4-32k:latest` | `gemma4-32k:latest` |
| `high-48gb+` | `2–4` *(min(4, cores//2))* | `2` | `65536` | `qwen2.5-coder:14b` | `gemma4-32k:latest` |

### Customization Knobs (`install_plugin.sh` Flags & Environment Variables)

You can inspect your machine's detected profile anytime with `./ollama-worker/.venv/bin/python ollama-worker/hardware.py --json` (or via the `get_worker_scaling_profile` MCP tool), and override any dimension via `install_plugin.sh` flags or `OLLAMA_WORKER_*` environment variables:

| Environment Variable | `install_plugin.sh` Flag | Effect |
| :--- | :--- | :--- |
| `OLLAMA_WORKER_MAX_PARALLEL` | `--max-parallel <N>` | Explicit max concurrent worker slots (`>= 1`; overrides auto/core cap) |
| `OLLAMA_WORKER_MAX_CORES` | `--max-cores <N>` | Overrides effective CPU cores and caps auto-computed parallelism |
| `OLLAMA_WORKER_MEMORY_GB` | `--memory-gb <GB>` | Overrides detected RAM/VRAM (useful in VMs/containers) |
| `OLLAMA_WORKER_CONTEXT_LENGTH` | `--context-length <N>` | Overrides recommended `OLLAMA_CONTEXT_LENGTH` (`>= 4096`) |
| `OLLAMA_WORKER_MODEL` | `--model <tag>` | Default single-worker model tag |
| `OLLAMA_WORKER_PARALLEL_MODEL` | `--parallel-model <tag>` | Recommended model tag when running parallel subagents |
| `OLLAMA_WORKER_LOCK_DIR` | *(env only)* | Custom directory for cross-process `slot-*.lock` files |
| `OLLAMA_WORKER_DISABLE_SLOT_LOCK` | *(env only)* | Set `1` to disable `fcntl.flock` and use per-process `asyncio.Semaphore` |

### Important Scaling Warnings

- **16 GB default is `max_parallel_workers=1` (`gemma4-32k:latest`):** `gemma4-32k:latest` provides reliable native OpenAI `tool_calls` and stays `100% GPU` single-stream. Avoid `qwen2.5-coder:7b` for agentic loops — it frequently emits raw JSON text instead of structured `tool_calls` (flagged as `status=tool_call_error` by `provider.py`).
- **KV cache multiplies with `OLLAMA_NUM_PARALLEL`:** Opting into `OLLAMA_NUM_PARALLEL=2` allocates `2x` KV cache upfront. On 16 GB machines, use a `<= 5 GB` native tool-calling model (`qwen3:4b` or `gemma4:e4b` via `--parallel-model`) for 2 concurrent subagents; running 2 slots of a `9 GB+` model risks CPU/swap spill.
- **Avoid model thrashing (`OLLAMA_MAX_LOADED_MODELS=1`):** On `<= 16 GB` hosts, keep concurrent subagents on the same model tag. Requesting two different models concurrently forces Ollama to evict and reload weights from disk.
- **Daemon defaults to `OLLAMA_NUM_PARALLEL=1`:** Unless tuned and restarted, Ollama queues concurrent requests serially even when `WorkerSlotGate` allows 2 slots.
- **Workspace write collisions:** Running parallel workers on the same `workspace_path` triggers a collision warning; give each subagent disjoint files or separate `git worktree` paths.

### Applying Daemon Tuning

- [ ] **Recommended (dynamic per host):** Preview or apply host-tuned daemon variables with `install_plugin.sh`:
  ```zsh
  # Dry-run (prints host profile + recommended launchctl/export commands):
  ./install_plugin.sh

  # Opt-in 2-slot parallel subagent tuning on 16 GB macOS (using a <=5 GB native tool-calling model):
  ./install_plugin.sh --max-parallel 2 --parallel-model qwen3:4b --tune-ollama
  ```
- [ ] *(Manual macOS `launchctl` example for 16 GB M1 with 2 parallel subagent slots)*:
  ```zsh
  launchctl setenv OLLAMA_NUM_PARALLEL 2
  launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
  launchctl setenv OLLAMA_CONTEXT_LENGTH 32768
  launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
  launchctl setenv OLLAMA_FLASH_ATTENTION 1
  ```
- [ ] *(Optional, if launching `ollama serve` manually from Terminal)* Add equivalent exports to `~/.zshrc` or `~/.bashrc`.
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

## 3. Installing as a Global Antigravity Plugin (Recommended)

Use [install_plugin.sh](install_plugin.sh) so `.agents/` and `plugin/` stay the single source of truth in Git while generating concrete local paths in `~/.gemini/config/plugins/ollama-worker/`:

- [ ] Run the idempotent plugin installer from the repository root:
  ```zsh
  ./install_plugin.sh
  ```
  This syncs `plugin/plugin.json`, `.agents/skills/radical-spec-delegate/SKILL.md`, `rules/AGENTS.md` (frontmatter stripped from `.agents/rules/quota-delegation.md`), and generates `mcp_config.json` from [mcp_config.snippet.json](ollama-worker/mcp_config.snippet.json).
- [ ] In Antigravity IDE/App, open **Customizations** (or **MCP Servers** > **Refresh**; in `agy` CLI run `/mcp`) and confirm `ollama-worker` is enabled and exposes `list_local_ollama_models`, `delegate_coding_task`, resource `ollama://models`, and prompt `radical_spec`.

### Alternative: Standalone `mcp_config.json` or Optional `mcp-harness` CLI

- [ ] *(Standalone `mcp_config.json`)* Preview resolved JSON with:
  ```zsh
  sed "s|/path/to/antigravity-ollama-worker|$PWD|g" ollama-worker/mcp_config.snippet.json
  ```
  Merge into `~/.gemini/config/mcp_config.json` (macOS/Linux), `%USERPROFILE%\.gemini\config\mcp_config.json` (Windows), or `.agents/mcp_config.json` (workspace-only).
- [ ] *(Optional — if you bring your own `mcp-harness` package)* Launch `OllamaWorkerProvider` directly via [harness.yml](ollama-worker/harness.yml):
  ```zsh
  PYTHONPATH="$PWD/ollama-worker" \
  "$PWD/ollama-worker/.venv/bin/python" \
  -m mcp_harness -c "$PWD/ollama-worker/harness.yml"
  ```

---

## 4. Verification Tests

- [ ] **Offline Protocol, Fallback & Policy Check:**
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

## 5. Troubleshooting Quick Checks

- **Config drift after editing `.agents/skills/` or `.agents/rules/`:**
  Re-run `./install_plugin.sh` so `~/.gemini/config/plugins/ollama-worker/` picks up the latest repo changes.
- **Stdio JSON framing errors / MCP disconnect on startup:**
  Ensure nothing prints to `stdout` in `server.py`. Confirm `logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)` stays at the very top of `ollama-worker/server.py` before any `google.antigravity`, `mcp`, or `mcp_harness` imports.
- **Model forgets instructions or truncates tool calls:**
  Run `ollama ps` during a task and check `CONTEXT`. If it shows `4096` or `8192` instead of `32768`, `launchctl setenv` was not picked up—fully quit Ollama (`pkill -x Ollama`) and relaunch it.
- **Slow generation / CPU-GPU offload split (`CPU/GPU` split in `ollama ps`):**
  The loaded model + `32768` KV cache exceeds available VRAM. Switch to a smaller model (e.g., `qwen2.5-coder:7b` or `gemma4-32k:latest`), or lower `OLLAMA_CONTEXT_LENGTH` to `16384`.
- **Stale `.venv` or wrong Python path:**
  Verify `./ollama-worker/.venv/bin/python --version` works and matches the `"command"` path in your generated `mcp_config.json`.
