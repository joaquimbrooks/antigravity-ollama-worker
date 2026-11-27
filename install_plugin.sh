#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="${HOME}/.gemini/config/plugins/ollama-worker"

MAX_PARALLEL=""
MAX_CORES=""
MEMORY_GB=""
CONTEXT_LENGTH=""
MODEL=""
PARALLEL_MODEL=""
TUNE_OLLAMA="false"

usage() {
  cat <<'EOF'
Usage: ./install_plugin.sh [options]

Options:
  --target <dir>            Plugin destination (default: ~/.gemini/config/plugins/ollama-worker)
  --max-parallel <N>        Override OLLAMA_WORKER_MAX_PARALLEL (>= 1)
  --max-cores <N>           Override OLLAMA_WORKER_MAX_CORES (>= 1)
  --memory-gb <GB>          Override OLLAMA_WORKER_MEMORY_GB (> 0)
  --context-length <N>      Override OLLAMA_WORKER_CONTEXT_LENGTH (>= 4096)
  --model <tag>             Override OLLAMA_WORKER_MODEL (single worker default)
  --parallel-model <tag>    Override OLLAMA_WORKER_PARALLEL_MODEL (concurrent subagents)
  --tune-ollama             Apply recommended OLLAMA_* daemon settings (launchctl on macOS)
  -h, --help                Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      [[ $# -ge 2 ]] || { echo "Error: --target requires a value." >&2; exit 1; }
      TARGET_DIR="$2"
      shift 2
      ;;
    --max-parallel)
      [[ $# -ge 2 ]] || { echo "Error: --max-parallel requires a value." >&2; exit 1; }
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --max-cores)
      [[ $# -ge 2 ]] || { echo "Error: --max-cores requires a value." >&2; exit 1; }
      MAX_CORES="$2"
      shift 2
      ;;
    --memory-gb)
      [[ $# -ge 2 ]] || { echo "Error: --memory-gb requires a value." >&2; exit 1; }
      MEMORY_GB="$2"
      shift 2
      ;;
    --context-length)
      [[ $# -ge 2 ]] || { echo "Error: --context-length requires a value." >&2; exit 1; }
      CONTEXT_LENGTH="$2"
      shift 2
      ;;
    --model)
      [[ $# -ge 2 ]] || { echo "Error: --model requires a value." >&2; exit 1; }
      MODEL="$2"
      shift 2
      ;;
    --parallel-model)
      [[ $# -ge 2 ]] || { echo "Error: --parallel-model requires a value." >&2; exit 1; }
      PARALLEL_MODEL="$2"
      shift 2
      ;;
    --tune-ollama)
      TUNE_OLLAMA="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

VENV_PYTHON="${REPO_ROOT}/ollama-worker/.venv/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Warning: ${VENV_PYTHON} not found or not executable yet." >&2
  echo "Create the venv first before enabling the MCP server in Antigravity." >&2
fi

mkdir -p "${TARGET_DIR}/skills/radical-spec-delegate" "${TARGET_DIR}/rules"

cp "${REPO_ROOT}/plugin/plugin.json" "${TARGET_DIR}/plugin.json"
cp "${REPO_ROOT}/plugin/README.md" "${TARGET_DIR}/README.md"
cp "${REPO_ROOT}/.agents/skills/radical-spec-delegate/SKILL.md" \
   "${TARGET_DIR}/skills/radical-spec-delegate/SKILL.md"

# Strip leading YAML frontmatter from .agents/rules/quota-delegation.md because
# plugin rules/AGENTS.md expects plain markdown without frontmatter.
python3 - "${REPO_ROOT}/.agents/rules/quota-delegation.md" "${TARGET_DIR}/rules/AGENTS.md" <<'PY'
import pathlib
import sys

src = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
lines = src.splitlines(keepends=True)
if lines and lines[0].strip() == "---":
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            lines = lines[idx + 1 :]
            break
text = "".join(lines).lstrip("\r\n")
pathlib.Path(sys.argv[2]).write_text(text, encoding="utf-8")
PY

# Compute host-aware scaling profile with optional CLI overrides and render mcp_config.json.
PROFILE_JSON="$(
  OLLAMA_WORKER_MAX_PARALLEL="${MAX_PARALLEL}" \
  OLLAMA_WORKER_MAX_CORES="${MAX_CORES}" \
  OLLAMA_WORKER_MEMORY_GB="${MEMORY_GB}" \
  OLLAMA_WORKER_CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
  OLLAMA_WORKER_MODEL="${MODEL}" \
  OLLAMA_WORKER_PARALLEL_MODEL="${PARALLEL_MODEL}" \
  python3 "${REPO_ROOT}/ollama-worker/hardware.py" --json
)"

PROFILE_JSON="${PROFILE_JSON}" \
MAX_PARALLEL="${MAX_PARALLEL}" \
MAX_CORES="${MAX_CORES}" \
MEMORY_GB="${MEMORY_GB}" \
CONTEXT_LENGTH="${CONTEXT_LENGTH}" \
MODEL="${MODEL}" \
PARALLEL_MODEL="${PARALLEL_MODEL}" \
python3 - "${REPO_ROOT}/ollama-worker/mcp_config.snippet.json" "${TARGET_DIR}/mcp_config.json" "${REPO_ROOT}" <<'PY'
import json
import os
import pathlib
import sys

src_path, dst_path, repo_root = sys.argv[1], sys.argv[2], sys.argv[3]
profile = json.loads(os.environ["PROFILE_JSON"])
raw = pathlib.Path(src_path).read_text(encoding="utf-8")
rendered = raw.replace("/path/to/antigravity-ollama-worker", repo_root)
parsed = json.loads(rendered)

server_cfg = parsed.setdefault("mcpServers", {}).setdefault("ollama-worker", {})
env_map = dict(server_cfg.get("env") or {})

# Always record explicit CLI overrides; also persist resolved concurrency so
# runtime behavior is deterministic and inspectable in mcp_config.json.
explicit_map = {
    "OLLAMA_WORKER_MAX_PARALLEL": os.environ.get("MAX_PARALLEL", "").strip(),
    "OLLAMA_WORKER_MAX_CORES": os.environ.get("MAX_CORES", "").strip(),
    "OLLAMA_WORKER_MEMORY_GB": os.environ.get("MEMORY_GB", "").strip(),
    "OLLAMA_WORKER_CONTEXT_LENGTH": os.environ.get("CONTEXT_LENGTH", "").strip(),
    "OLLAMA_WORKER_MODEL": os.environ.get("MODEL", "").strip(),
    "OLLAMA_WORKER_PARALLEL_MODEL": os.environ.get("PARALLEL_MODEL", "").strip(),
}
for key, val in explicit_map.items():
    if val:
        env_map[key] = val

if env_map:
    server_cfg["env"] = env_map

pathlib.Path(dst_path).write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
PY

eval "$(
  PROFILE_JSON="${PROFILE_JSON}" python3 - <<'PY'
import json
import os
import shlex

p = json.loads(os.environ["PROFILE_JSON"])
print(f"DETECTED_TIER={shlex.quote(str(p['tier']))}")
print(f"DETECTED_CORES={shlex.quote(str(p['effective_cores']))}")
print(f"DETECTED_MEM_GB={shlex.quote(str(p['effective_memory_gb']))}")
print(f"REC_PARALLEL={shlex.quote(str(p['recommended_ollama_num_parallel']))}")
print(f"REC_MAX_LOADED={shlex.quote(str(p['recommended_ollama_max_loaded_models']))}")
print(f"REC_CTX={shlex.quote(str(p['recommended_context_length']))}")
print(f"REC_KV={shlex.quote(str(p['recommended_kv_cache_type']))}")
print(f"REC_FLASH={shlex.quote(str(p['recommended_flash_attention']))}")
print(f"REC_SINGLE_MODEL={shlex.quote(str(p['recommended_single_model']))}")
print(f"REC_PARALLEL_MODEL={shlex.quote(str(p['recommended_parallel_model']))}")
PY
)"

echo "Installed ollama-worker plugin to: ${TARGET_DIR}"
echo "  - plugin.json"
echo "  - mcp_config.json (repo_root=${REPO_ROOT}, max_parallel=${REC_PARALLEL})"
echo "  - skills/radical-spec-delegate/SKILL.md"
echo "  - rules/AGENTS.md"
echo ""
echo "Host Scaling Summary:"
echo "  - Tier: ${DETECTED_TIER} (effective_cores=${DETECTED_CORES}, effective_memory=${DETECTED_MEM_GB} GB)"
echo "  - Models: single=${REC_SINGLE_MODEL} | parallel_subagents=${REC_PARALLEL_MODEL}"

PROFILE_JSON="${PROFILE_JSON}" python3 - <<'PY'
import json
import os

warnings = json.loads(os.environ["PROFILE_JSON"]).get("warnings") or []
if warnings:
    print("  - Warnings:")
    for w in warnings:
        print(f"      * {w}")
PY

OS_NAME="$(uname -s)"
echo ""
if [[ "${TUNE_OLLAMA}" == "true" ]]; then
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    echo "Applying Ollama daemon settings via launchctl setenv..."
    launchctl setenv OLLAMA_NUM_PARALLEL "${REC_PARALLEL}"
    launchctl setenv OLLAMA_MAX_LOADED_MODELS "${REC_MAX_LOADED}"
    launchctl setenv OLLAMA_CONTEXT_LENGTH "${REC_CTX}"
    launchctl setenv OLLAMA_KV_CACHE_TYPE "${REC_KV}"
    launchctl setenv OLLAMA_FLASH_ATTENTION "${REC_FLASH}"
    echo "Applied launchctl environment variables. Restart Ollama for changes to take effect:"
    echo "  pkill -x Ollama || true; pkill -x ollama || true; open -a Ollama"
  else
    echo "Non-macOS host detected (${OS_NAME}). Add these environment variables to your shell or systemd service:"
    echo "  export OLLAMA_NUM_PARALLEL=${REC_PARALLEL}"
    echo "  export OLLAMA_MAX_LOADED_MODELS=${REC_MAX_LOADED}"
    echo "  export OLLAMA_CONTEXT_LENGTH=${REC_CTX}"
    echo "  export OLLAMA_KV_CACHE_TYPE=${REC_KV}"
    echo "  export OLLAMA_FLASH_ATTENTION=${REC_FLASH}"
  fi
else
  echo "Recommended Ollama Daemon Tuning (dry-run; pass --tune-ollama to apply on macOS):"
  if [[ "${OS_NAME}" == "Darwin" ]]; then
    echo "  launchctl setenv OLLAMA_NUM_PARALLEL ${REC_PARALLEL}"
    echo "  launchctl setenv OLLAMA_MAX_LOADED_MODELS ${REC_MAX_LOADED}"
    echo "  launchctl setenv OLLAMA_CONTEXT_LENGTH ${REC_CTX}"
    echo "  launchctl setenv OLLAMA_KV_CACHE_TYPE ${REC_KV}"
    echo "  launchctl setenv OLLAMA_FLASH_ATTENTION ${REC_FLASH}"
    echo "  pkill -x Ollama || true; pkill -x ollama || true; open -a Ollama"
  else
    echo "  export OLLAMA_NUM_PARALLEL=${REC_PARALLEL}"
    echo "  export OLLAMA_MAX_LOADED_MODELS=${REC_MAX_LOADED}"
    echo "  export OLLAMA_CONTEXT_LENGTH=${REC_CTX}"
    echo "  export OLLAMA_KV_CACHE_TYPE=${REC_KV}"
    echo "  export OLLAMA_FLASH_ATTENTION=${REC_FLASH}"
  fi
fi
