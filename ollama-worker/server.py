"""Local Ollama coding delegation worker exposed as a native stdio MCP server."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Critical: configure all logging to stderr before importing or initializing
# mcp_harness or google-antigravity so logs never corrupt stdio JSON framing.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)

# Ensure local sibling modules (policies.py, provider.py) resolve regardless of CWD.
_WORKER_DIR = str(Path(__file__).resolve().parent)
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from mcp_harness.server import MCPHarness  # noqa: E402

from policies import (  # noqa: E402
    BLOCKED_COMMAND_PATTERNS,
    build_worker_policies,
    is_blocked_command,
)
from provider import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OLLAMA_TAGS_URL,
    OllamaWorkerProvider,
)


def create_harness() -> MCPHarness:
    """Build the configured MCPHarness instance with middleware and OllamaWorkerProvider."""
    harness = MCPHarness(
        server_name="ollama-worker",
        version="0.1.0",
        instructions=(
            "Local Ollama coding delegation worker. Use list_local_ollama_models to "
            "inspect installed models and delegate_coding_task with a self-contained "
            "Radical Spec."
        ),
    )
    harness.use_logging(log_level=logging.WARNING)
    harness.use_metrics()
    harness.register_provider(OllamaWorkerProvider())
    return harness


harness = create_harness()


if __name__ == "__main__":
    harness.run_stdio(log_level=logging.WARNING)


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_BASE_URL",
    "OLLAMA_TAGS_URL",
    "BLOCKED_COMMAND_PATTERNS",
    "is_blocked_command",
    "build_worker_policies",
    "OllamaWorkerProvider",
    "create_harness",
    "harness",
]
