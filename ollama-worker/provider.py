"""Local Ollama coding delegation core and optional mcp-harness BaseProvider."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

from google.antigravity import Agent, LocalOpenAIAgentConfig

from policies import build_worker_policies

try:
    from mcp_harness.providers.base import (
        BaseProvider,
        PromptArgument,
        prompt,
        resource,
        tool,
    )

    HAS_MCP_HARNESS = True
except ImportError:
    BaseProvider = None  # type: ignore[assignment,misc]
    PromptArgument = None  # type: ignore[assignment,misc]
    HAS_MCP_HARNESS = False

DEFAULT_MODEL = "gemma4-32k:latest"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"


def _format_size_gb(size_bytes: int | float | None) -> str:
    """Format byte size into a human-readable GB string."""
    if not size_bytes:
        return "unknown size"
    gb = float(size_bytes) / (1024**3)
    return f"{gb:.2f} GB"


class OllamaWorkerCore:
    """Framework-agnostic Ollama delegation engine shared by MCPServer and mcp-harness."""

    def __init__(
        self,
        default_model: str | None = None,
        default_base_url: str | None = None,
        ollama_tags_url: str | None = None,
    ) -> None:
        self.default_model = (
            default_model or os.environ.get("OLLAMA_WORKER_MODEL") or DEFAULT_MODEL
        )
        self.default_base_url = (
            default_base_url or os.environ.get("OLLAMA_WORKER_BASE_URL") or DEFAULT_BASE_URL
        )
        self.ollama_tags_url = (
            ollama_tags_url or os.environ.get("OLLAMA_TAGS_URL") or OLLAMA_TAGS_URL
        )

    def fetch_ollama_tags(self) -> dict[str, Any]:
        """Query the local Ollama /api/tags endpoint and return parsed JSON."""
        req = urllib.request.Request(self.ollama_tags_url, method="GET")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return payload

    def list_local_ollama_models(self) -> str:
        """Query the local Ollama server and return installed model tags and sizes."""
        try:
            payload = self.fetch_ollama_tags()
        except urllib.error.URLError as exc:
            return (
                f"Error: Could not reach local Ollama at {self.ollama_tags_url} ({exc.reason}). "
                "Ensure Ollama is running (e.g. open -a Ollama or `ollama serve`)."
            )
        except Exception as exc:  # noqa: BLE001
            return f"Error querying local Ollama models: {type(exc).__name__}: {exc}"

        models = payload.get("models") or []
        if not models:
            return (
                "Ollama is running, but no local models are installed. "
                "Pull a tool-capable model first (e.g. `ollama pull qwen3:14b`)."
            )

        lines = [f"Installed local Ollama models ({len(models)}):"]
        for entry in models:
            model_name = entry.get("name", "<unknown>")
            size_str = _format_size_gb(entry.get("size"))
            details = entry.get("details") or {}
            param_size = details.get("parameter_size", "")
            quant = details.get("quantization_level", "")
            meta = ", ".join(part for part in (param_size, quant) if part)
            suffix = f" [{meta}]" if meta else ""
            lines.append(f"- {model_name} ({size_str}){suffix}")

        return "\n".join(lines)

    async def delegate_coding_task(
        self,
        spec: str,
        workspace_path: str,
        model: str = "",
        base_url: str = "",
    ) -> str:
        """Delegate a self-contained Radical Spec coding task to a local Ollama model.

        Args:
        spec: Complete, self-contained Radical Spec (objective, file scope, contracts, ordered steps, and verification command).
        workspace_path: Absolute path to the target project workspace directory.
        model: Local Ollama model tag (e.g. 'gemma4-32k:latest', 'qwen2.5-coder:7b'). Leave empty to use provider default.
        base_url: Local OpenAI-compatible Ollama endpoint URL. Leave empty to use provider default.
        """
        if not spec or not spec.strip():
            return "Error: 'spec' must be a non-empty Radical Spec."

        if not workspace_path or not os.path.isabs(workspace_path):
            return (
                f"Error: 'workspace_path' must be an absolute directory path, "
                f"got: {workspace_path!r}"
            )

        if not os.path.isdir(workspace_path):
            return (
                f"Error: workspace_path does not exist or is not a directory: "
                f"{workspace_path}"
            )

        resolved_model = (model or "").strip() or self.default_model
        resolved_base_url = (base_url or "").strip() or self.default_base_url

        config = LocalOpenAIAgentConfig(
            model=resolved_model,
            base_url=resolved_base_url,
            workspaces=[workspace_path],
            policies=build_worker_policies(),
        ).lightweight()

        output_chunks: list[str] = []
        status = "completed"
        error_detail = ""

        try:
            async with Agent(config) as agent:
                response = await agent.chat(spec)
                async for token in response:
                    output_chunks.append(token)
        except asyncio.CancelledError:
            status = "cancelled"
            error_detail = "Task was cancelled before completion."
        except Exception as exc:  # noqa: BLE001
            status = "error"
            error_detail = f"{type(exc).__name__}: {exc}"
        finally:
            partial_output = "".join(output_chunks).strip()

        if status == "completed":
            body = partial_output or "(No text output returned by local worker.)"
            return (
                f"[ollama-worker | model={resolved_model} | workspace={workspace_path}]\n\n"
                f"{body}\n\n"
                "---\n"
                "Next step: Inspect `git diff` and `git status` in the workspace to verify "
                "all local changes before accepting."
            )

        lines = [
            f"[ollama-worker | status={status} | model={resolved_model} | workspace={workspace_path}]",
            f"Diagnostic: {error_detail}",
        ]
        if partial_output:
            lines.append(f"\nPartial output before {status}:\n{partial_output}")
        lines.append(
            "\nNext step: Inspect `git diff` / `git status` if the worker made partial edits."
        )
        return "\n".join(lines)

    def models_snapshot_json(self) -> str:
        """Return installed Ollama models (or connection diagnostic) serialized as JSON."""
        try:
            payload = self.fetch_ollama_tags()
            data: dict[str, Any] = {
                "endpoint": self.ollama_tags_url,
                "default_model": self.default_model,
                "models": payload.get("models") or [],
            }
        except Exception as exc:  # noqa: BLE001
            data = {
                "endpoint": self.ollama_tags_url,
                "default_model": self.default_model,
                "error": f"{type(exc).__name__}: {exc}",
                "models": [],
            }
        return json.dumps(data, indent=2)

    @staticmethod
    def build_radical_spec_text(
        objective: str,
        workspace_path: str,
        verification_command: str = "pytest -q",
    ) -> str:
        """Render a structured Radical Spec prompt body."""
        return (
            f"# Radical Spec\n\n"
            f"## Objective\n{objective}\n\n"
            f"## Workspace\n`{workspace_path}`\n\n"
            f"## File Scope & Do-Not-Touch List\n"
            f"- Edit only the files explicitly required for the objective.\n"
            f"- Do not modify unrelated modules or global configs.\n\n"
            f"## Contracts & Invariants\n"
            f"- Preserve public signatures, type annotations, and existing error handling.\n\n"
            f"## Ordered Steps\n"
            f"1. Inspect the relevant interfaces and tests.\n"
            f"2. Implement the requested changes cleanly.\n"
            f"3. Run the verification command and fix any failures.\n\n"
            f"## Verification Command\n```bash\n{verification_command}\n```\n\n"
            f"## Return Format\n"
            f"Summarize files changed, commands run, and verification output."
        )


if HAS_MCP_HARNESS:

    class OllamaWorkerProvider(BaseProvider):  # type: ignore[valid-type,misc]
        """mcp-harness BaseProvider wrapping OllamaWorkerCore."""

        def __init__(
            self,
            default_model: str | None = None,
            default_base_url: str | None = None,
            ollama_tags_url: str | None = None,
        ) -> None:
            self.core = OllamaWorkerCore(
                default_model=default_model,
                default_base_url=default_base_url,
                ollama_tags_url=ollama_tags_url,
            )

        @property
        def name(self) -> str:
            return "ollama-worker"

        @tool(
            name="list_local_ollama_models",
            description="Query the local Ollama server and return installed model tags and sizes.",
        )
        def list_local_ollama_models(self) -> str:
            """Query the local Ollama server and return installed model tags and sizes."""
            return self.core.list_local_ollama_models()

        @tool(
            name="delegate_coding_task",
            description="Delegate a self-contained Radical Spec coding task to a local Ollama model.",
        )
        async def delegate_coding_task(
            self,
            spec: str,
            workspace_path: str,
            model: str = "",
            base_url: str = "",
        ) -> str:
            """Delegate a self-contained Radical Spec coding task to a local Ollama model.

            Args:
            spec: Complete, self-contained Radical Spec (objective, file scope, contracts, ordered steps, and verification command).
            workspace_path: Absolute path to the target project workspace directory.
            model: Local Ollama model tag (e.g. 'gemma4-32k:latest', 'qwen2.5-coder:7b'). Leave empty to use provider default.
            base_url: Local OpenAI-compatible Ollama endpoint URL. Leave empty to use provider default.
            """
            return await self.core.delegate_coding_task(
                spec=spec,
                workspace_path=workspace_path,
                model=model,
                base_url=base_url,
            )

        @resource(
            uri_pattern="ollama://models",
            name="Local Ollama Models",
            description="Read-only JSON snapshot of installed local Ollama models.",
            mime_type="application/json",
        )
        def read_models_resource(self, uri: str) -> dict[str, Any]:
            """Return installed Ollama models as an MCP JSON resource payload."""
            return {
                "uri": uri,
                "mimeType": "application/json",
                "text": self.core.models_snapshot_json(),
            }

        @prompt(
            name="radical_spec",
            description="Generate a structured self-contained Radical Spec prompt for local Ollama coding delegation.",
            arguments=[
                PromptArgument(
                    name="objective",
                    description="Desired behavior, refactor, or bug fix.",
                    required=True,
                ),
                PromptArgument(
                    name="workspace_path",
                    description="Absolute path to the target project workspace.",
                    required=True,
                ),
                PromptArgument(
                    name="verification_command",
                    description="Exact verification command to run after changes.",
                    required=False,
                ),
            ],
        )
        def radical_spec_prompt(
            self,
            objective: str,
            workspace_path: str,
            verification_command: str = "pytest -q",
        ) -> dict[str, Any]:
            """Return a structured Radical Spec prompt message for local worker delegation."""
            template = self.core.build_radical_spec_text(
                objective=objective,
                workspace_path=workspace_path,
                verification_command=verification_command,
            )
            return {
                "description": "Structured Radical Spec prompt for local Ollama delegation.",
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": template,
                        },
                    }
                ],
            }

else:
    OllamaWorkerProvider = None  # type: ignore[assignment,misc]


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_BASE_URL",
    "OLLAMA_TAGS_URL",
    "HAS_MCP_HARNESS",
    "OllamaWorkerCore",
    "OllamaWorkerProvider",
]
