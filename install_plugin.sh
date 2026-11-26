#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="${HOME}/.gemini/config/plugins/ollama-worker"

if [[ "${1:-}" == "--target" && -n "${2:-}" ]]; then
  TARGET_DIR="$2"
fi

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

# Generate mcp_config.json with concrete local repo paths from the template.
python3 - "${REPO_ROOT}/ollama-worker/mcp_config.snippet.json" "${TARGET_DIR}/mcp_config.json" "${REPO_ROOT}" <<'PY'
import json
import pathlib
import sys

src_path, dst_path, repo_root = sys.argv[1], sys.argv[2], sys.argv[3]
raw = pathlib.Path(src_path).read_text(encoding="utf-8")
rendered = raw.replace("/path/to/antigravity-ollama-worker", repo_root)
parsed = json.loads(rendered)
pathlib.Path(dst_path).write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
PY

echo "Installed ollama-worker plugin to: ${TARGET_DIR}"
echo "  - plugin.json"
echo "  - mcp_config.json (repo_root=${REPO_ROOT})"
echo "  - skills/radical-spec-delegate/SKILL.md"
echo "  - rules/AGENTS.md"
