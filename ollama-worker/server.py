"""Local Ollama coding delegation worker exposed as a native stdio MCP server.

Uses mcp-harness (MCPHarness + OllamaWorkerProvider + middleware) when installed,
and falls back cleanly to the standard `mcp` SDK (`MCPServer`) included with
`google-antigravity` when `mcp-harness` is absent.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Critical: configure all logging to stderr before importing or initializing
# mcp_harness, mcp, or google-antigravity so logs never corrupt stdio JSON framing.
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)

# Ensure local sibling modules (policies.py, provider.py) resolve regardless of CWD.
_WORKER_DIR = str(Path(__file__).resolve().parent)
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from mcp.server.mcpserver import MCPServer  # noqa: E402

from policies import (  # noqa: E402
    BLOCKED_COMMAND_PATTERNS,
    build_worker_policies,
    is_blocked_command,
)
from provider import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    HAS_MCP_HARNESS,
    OLLAMA_TAGS_URL,
    OllamaWorkerCore,
    OllamaWorkerProvider,
)

if HAS_MCP_HARNESS:
    from mcp_harness.server import MCPHarness  # noqa: E402
else:
    MCPHarness = None  # type: ignore[assignment,misc]

SERVER_NAME = "ollama-worker"
SERVER_INSTRUCTIONS = (
    "Local Ollama coding delegation worker. Use list_local_ollama_models to "
    "inspect installed models and delegate_coding_task with a self-contained "
    "Radical Spec."
)


def create_harness() -> Any:
    """Build the configured MCPHarness instance (requires optional mcp-harness)."""
    if not HAS_MCP_HARNESS or MCPHarness is None or OllamaWorkerProvider is None:
        raise ImportError(
            "mcp-harness is not installed. Install with `[harness]` optional extra "
            "or run server.py via the default MCPServer fallback."
        )

    harness_instance = MCPHarness(
        server_name=SERVER_NAME,
        version="0.1.0",
        instructions=SERVER_INSTRUCTIONS,
    )
    harness_instance.use_logging(log_level=logging.WARNING)
    harness_instance.use_metrics()
    harness_instance.register_provider(OllamaWorkerProvider())
    return harness_instance


def create_mcpserver_fallback(core: OllamaWorkerCore | None = None) -> MCPServer:
    """Build a standalone `mcp.server.mcpserver.MCPServer` without mcp-harness."""
    worker_core = core or OllamaWorkerCore()
    mcp_server = MCPServer(SERVER_NAME, log_level="WARNING")

    @mcp_server.tool()
    def list_local_ollama_models() -> str:
        """Query the local Ollama server and return installed model tags and sizes."""
        return worker_core.list_local_ollama_models()

    @mcp_server.tool()
    async def delegate_coding_task(
        spec: str,
        workspace_path: str,
        model: str = "",
        base_url: str = "",
    ) -> str:
        """Delegate a self-contained Radical Spec coding task to a local Ollama model.

        Args:
            spec: Complete, self-contained Radical Spec (objective, file scope,
                contracts, ordered steps, and verification command).
            workspace_path: Absolute path to the target project workspace directory.
            model: Local Ollama model tag (e.g. 'gemma4-32k:latest', 'qwen2.5-coder:7b').
                Leave empty to use the worker default.
            base_url: Local OpenAI-compatible Ollama endpoint URL. Leave empty to use
                the worker default.
        """
        return await worker_core.delegate_coding_task(
            spec=spec,
            workspace_path=workspace_path,
            model=model,
            base_url=base_url,
        )

    @mcp_server.resource(
        "ollama://models",
        name="Local Ollama Models",
        description="Read-only JSON snapshot of installed local Ollama models.",
        mime_type="application/json",
    )
    def read_models_resource() -> str:
        """Return installed Ollama models as an MCP JSON resource payload."""
        return worker_core.models_snapshot_json()

    @mcp_server.prompt(
        name="radical_spec",
        description=(
            "Generate a structured self-contained Radical Spec prompt for local "
            "Ollama coding delegation."
        ),
    )
    def radical_spec_prompt(
        objective: str,
        workspace_path: str,
        verification_command: str = "pytest -q",
    ) -> str:
        """Return a structured Radical Spec prompt message for local worker delegation."""
        return worker_core.build_radical_spec_text(
            objective=objective,
            workspace_path=workspace_path,
            verification_command=verification_command,
        )

    return mcp_server


harness = create_harness() if HAS_MCP_HARNESS else None
mcp = create_mcpserver_fallback()


def run_stdio() -> None:
    """Run STDIO MCP server via mcp-harness when installed, else MCPServer fallback."""
    if harness is not None:
        harness.run_stdio(log_level=logging.WARNING)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    run_stdio()


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_BASE_URL",
    "OLLAMA_TAGS_URL",
    "HAS_MCP_HARNESS",
    "BLOCKED_COMMAND_PATTERNS",
    "is_blocked_command",
    "build_worker_policies",
    "OllamaWorkerCore",
    "OllamaWorkerProvider",
    "create_harness",
    "create_mcpserver_fallback",
    "harness",
    "mcp",
    "run_stdio",
]
