"""Local Ollama coding delegation core and optional mcp-harness BaseProvider."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable

from google.antigravity import Agent, LocalOpenAIAgentConfig, types

from hardware import (
    WorkerScalingProfile,
    WorkerSlotGate,
    build_scaling_profile,
)
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
DEFAULT_IDLE_TIMEOUT_SECONDS = 120.0
DEFAULT_SYNC_MAX_WALL_SECONDS = 165.0
DEFAULT_ASYNC_MAX_WALL_SECONDS = 600.0
LEAKED_CONTROL_TOKENS: tuple[str, ...] = (
    "<channel|>",
    "<|channel>",
    "<|call>",
    "<|tool_call>",
    "<|tool_call_start|>",
)
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_FENCED_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_-]*)\s*\n(.*?)```", re.DOTALL)


def _parse_positive_float_env(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in _TRUTHY_ENV_VALUES


def _format_size_gb(size_bytes: int | float | None) -> str:
    """Format byte size into a human-readable GB string."""
    if not size_bytes:
        return "unknown size"
    gb = float(size_bytes) / (1024**3)
    return f"{gb:.2f} GB"


def _is_raw_tool_call_item(item: Any) -> bool:
    """Return True if a JSON object matches common raw tool-call payload shapes."""
    if not isinstance(item, dict):
        return False
    if isinstance(item.get("name"), str) and any(
        key in item for key in ("arguments", "parameters", "args")
    ):
        return True
    fn = item.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str) and any(
        key in fn for key in ("arguments", "parameters", "args")
    ):
        return True
    return False


def _is_raw_tool_call_payload(parsed: Any) -> bool:
    """Return True if parsed JSON represents one or more unexecuted tool calls."""
    if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list) and parsed["tool_calls"]:
        return all(_is_raw_tool_call_item(item) for item in parsed["tool_calls"])
    items = parsed if isinstance(parsed, list) else [parsed]
    return bool(items) and all(_is_raw_tool_call_item(item) for item in items)


def _looks_like_raw_tool_call(text: str) -> bool:
    """Return True if worker output is an unexecuted raw JSON tool call emitted as text."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
    if not stripped or stripped[0] not in "{[":
        return False
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return _is_raw_tool_call_payload(parsed)


def _contains_raw_tool_call(text: str) -> bool:
    """Return True if output contains prose + a fenced or inline raw JSON tool call."""
    if _looks_like_raw_tool_call(text):
        return True
    if not text:
        return False
    for match in _FENCED_BLOCK_RE.finditer(text):
        candidate = match.group(1).strip()
        if not candidate or candidate[0] not in "{[":
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if _is_raw_tool_call_payload(parsed):
            return True

    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    while idx < length:
        ch = text[idx]
        if ch in "{[":
            try:
                parsed, end_idx = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                idx += 1
                continue
            if _is_raw_tool_call_payload(parsed):
                return True
            idx = max(idx + 1, end_idx)
        else:
            idx += 1
    return False


def _find_leaked_control_token(text: str) -> str | None:
    """Return the first leaked template control token found in plain text, if any."""
    content = text or ""
    for token in LEAKED_CONTROL_TOKENS:
        if token in content:
            return token
    return None


def _extract_harness_process(agent: Any) -> tuple[Any, Any]:
    """Best-effort extract (connection, subprocess.Popen) from an Agent instance."""
    conv = getattr(agent, "_conversation", None)
    conn = getattr(conv, "_connection", None) if conv is not None else None
    proc = getattr(conn, "_process", None) if conn is not None else None
    return conn, proc


def _force_kill_process(proc: Any) -> None:
    """Synchronously force-kill a still-running subprocess without raising."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:  # noqa: BLE001
        return
    stdin = getattr(proc, "stdin", None)
    if stdin is not None:
        with contextlib.suppress(Exception):
            stdin.close()
    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=1.0)


async def _force_abort_agent(agent: Any) -> bool:
    """Abort an in-flight Agent turn and terminate/kill its localharness process."""
    conv = getattr(agent, "_conversation", None)
    if conv is not None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(conv.cancel(), timeout=1.0)

    conn, proc = _extract_harness_process(agent)
    if conn is not None:
        with contextlib.suppress(Exception):
            conn._disconnecting = True  # noqa: SLF001

    if proc is None:
        return False
    try:
        if proc.poll() is not None:
            return False
    except Exception:  # noqa: BLE001
        return False

    stdin = getattr(proc, "stdin", None)
    if stdin is not None:
        with contextlib.suppress(Exception):
            stdin.close()
    with contextlib.suppress(Exception):
        proc.terminate()
    try:
        for _ in range(10):
            if proc.poll() is not None:
                return True
            await asyncio.sleep(0.05)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                await asyncio.to_thread(proc.wait, 1.0)
    except BaseException:
        _force_kill_process(proc)
        raise
    return True


class OllamaWorkerCore:
    """Framework-agnostic Ollama delegation engine shared by MCPServer and mcp-harness."""

    def __init__(
        self,
        default_model: str | None = None,
        default_base_url: str | None = None,
        ollama_tags_url: str | None = None,
    ) -> None:
        self.scaling_profile: WorkerScalingProfile = build_scaling_profile()
        self.slot_gate = WorkerSlotGate(
            max_slots=self.scaling_profile.max_parallel_workers
        )
        self.default_model = (
            default_model
            or os.environ.get("OLLAMA_WORKER_MODEL")
            or self.scaling_profile.recommended_single_model
            or DEFAULT_MODEL
        )
        self.default_base_url = (
            default_base_url or os.environ.get("OLLAMA_WORKER_BASE_URL") or DEFAULT_BASE_URL
        )
        self.ollama_tags_url = (
            ollama_tags_url or os.environ.get("OLLAMA_TAGS_URL") or OLLAMA_TAGS_URL
        )
        self._jobs: dict[str, dict[str, Any]] = {}
        self._job_tasks: dict[str, asyncio.Task[None]] = {}
        self._active_agents: dict[str, Any] = {}

    def fetch_ollama_tags(self) -> dict[str, Any]:
        """Query the local Ollama /api/tags endpoint and return parsed JSON."""
        req = urllib.request.Request(self.ollama_tags_url, method="GET")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return payload

    def get_worker_scaling_profile(self) -> str:
        """Return detected host hardware profile, concurrency limits, and warnings."""
        return self.scaling_profile.format_summary()

    def list_local_ollama_models(self) -> str:
        """Query the local Ollama server and return installed model tags and sizes."""
        try:
            payload = self.fetch_ollama_tags()
        except urllib.error.URLError as exc:
            return (
                f"Error: Could not reach local Ollama at {self.ollama_tags_url} ({exc.reason}). "
                "Ensure Ollama is running (e.g. open -a Ollama or `ollama serve`).\n\n"
                f"{self.scaling_profile.format_summary()}"
            )
        except Exception as exc:  # noqa: BLE001
            return (
                f"Error querying local Ollama models: {type(exc).__name__}: {exc}\n\n"
                f"{self.scaling_profile.format_summary()}"
            )

        models = payload.get("models") or []
        if not models:
            return (
                "Ollama is running, but no local models are installed. "
                "Pull a tool-capable model first (e.g. `ollama pull gemma4-32k:latest`).\n\n"
                f"{self.scaling_profile.format_summary()}"
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

        lines.append("")
        lines.append(self.scaling_profile.format_summary())
        return "\n".join(lines)

    @staticmethod
    def _validate_delegation_args(spec: str, workspace_path: str) -> str | None:
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
        return None

    async def _execute_agent(
        self,
        config: Any,
        spec: str,
        on_chunk: Callable[[str], None] | None = None,
        *,
        telemetry: dict[str, Any] | None = None,
        idle_timeout_seconds: float | None = None,
        max_wall_seconds: float | None = None,
        run_id: str | None = None,
    ) -> tuple[str, str, str]:
        """Run LocalOpenAIAgentConfig Agent loop and return (status, error_detail, partial_output)."""
        output_chunks: list[str] = []
        status = "completed"
        error_detail = ""
        started = time.monotonic()
        last_activity_at = started
        phase = "starting"
        thinking_chunks = 0
        tool_calls_executed = 0
        tool_errors = 0
        last_tool_name = ""
        turns = 0
        seen_model_steps: set[tuple[str, int]] = set()
        seen_active_tool_ids: set[str] = set()
        counted_done_tool_ids: set[str] = set()
        aborted_intentionally = False
        active_agent: Any = None

        idle_limit = (
            float(idle_timeout_seconds)
            if idle_timeout_seconds is not None and idle_timeout_seconds > 0
            else None
        )
        wall_limit = (
            float(max_wall_seconds)
            if max_wall_seconds is not None and max_wall_seconds > 0
            else None
        )

        def _touch(next_phase: str | None = None) -> None:
            nonlocal phase, last_activity_at
            last_activity_at = time.monotonic()
            if next_phase is not None:
                phase = next_phase
            if telemetry is not None:
                telemetry["phase"] = phase
                telemetry["thinking_chunks"] = thinking_chunks
                telemetry["tool_calls_executed"] = tool_calls_executed
                telemetry["tool_errors"] = tool_errors
                telemetry["last_tool_name"] = last_tool_name
                telemetry["turns"] = turns
                telemetry["last_activity_at"] = last_activity_at
                telemetry["elapsed_seconds"] = round(last_activity_at - started, 2)

        def _timeout_detail(now_ts: float) -> str:
            elapsed_s = round(now_ts - started, 1)
            idle_s = round(now_ts - last_activity_at, 1)
            wall_hit = wall_limit is not None and (now_ts - started) >= wall_limit
            kind = (
                f"Wall timeout exceeded ({wall_limit:.1f}s)"
                if wall_hit
                else f"Idle timeout exceeded ({idle_limit:.1f}s without activity)"
            )
            msg = (
                f"{kind} during phase={phase} "
                f"(elapsed={elapsed_s}s, idle={idle_s}s, chunks={len(output_chunks)}, "
                f"thinking_chunks={thinking_chunks}, tool_calls={tool_calls_executed}, "
                f"turns={turns}, last_tool={last_tool_name or 'none'})."
            )
            leaked = _find_leaked_control_token("".join(output_chunks))
            if leaked is not None:
                msg += f" Detected leaked raw control token {leaked!r} before stall."
            return msg

        try:
            async with Agent(config) as agent:
                active_agent = agent
                if run_id:
                    self._active_agents[run_id] = agent
                try:
                    _touch("waiting_llm")
                    await agent.conversation.send(spec)
                    step_iter = agent.conversation.receive_steps()
                    while True:
                        now = time.monotonic()
                        candidates: list[float] = []
                        if idle_limit is not None:
                            candidates.append(idle_limit - (now - last_activity_at))
                        if wall_limit is not None:
                            candidates.append(wall_limit - (now - started))
                        next_timeout = min(candidates) if candidates else None

                        if next_timeout is not None and next_timeout <= 0:
                            aborted_intentionally = True
                            status = "timeout"
                            error_detail = _timeout_detail(now)
                            with contextlib.suppress(BaseException):
                                await _force_abort_agent(agent)
                            break

                        try:
                            if next_timeout is not None:
                                step = await asyncio.wait_for(
                                    anext(step_iter), timeout=next_timeout
                                )
                            else:
                                step = await anext(step_iter)
                        except StopAsyncIteration:
                            break
                        except asyncio.TimeoutError:
                            aborted_intentionally = True
                            status = "timeout"
                            error_detail = _timeout_detail(time.monotonic())
                            with contextlib.suppress(BaseException):
                                await _force_abort_agent(agent)
                            break

                        if step.source == types.StepSource.MODEL:
                            step_key = (step.trajectory_id, step.step_index)
                            if step_key not in seen_model_steps:
                                seen_model_steps.add(step_key)
                                turns += 1

                        handled_special = False
                        if step.type == types.StepType.TOOL_CALL and step.tool_calls:
                            handled_special = True
                            for tc in step.tool_calls:
                                tc_key = tc.id or f"{step.id}:{tc.name}"
                                last_tool_name = tc.name
                                if step.status == types.StepStatus.ACTIVE:
                                    seen_active_tool_ids.add(tc_key)
                                    _touch("tool_running")
                                elif step.status in (
                                    types.StepStatus.DONE,
                                    types.StepStatus.ERROR,
                                ):
                                    if tc_key not in counted_done_tool_ids:
                                        counted_done_tool_ids.add(tc_key)
                                        tool_calls_executed += 1
                                        if step.status == types.StepStatus.ERROR:
                                            tool_errors += 1
                                    _touch("waiting_llm")
                                else:
                                    _touch("tool_running")

                        if step.thinking_delta:
                            handled_special = True
                            thinking_chunks += 1
                            _touch("thinking")

                        if (
                            step.content_delta
                            and step.source == types.StepSource.MODEL
                            and step.target == types.StepTarget.USER
                            and step.status != types.StepStatus.ERROR
                        ):
                            handled_special = True
                            output_chunks.append(step.content_delta)
                            _touch("streaming")
                            if on_chunk is not None:
                                on_chunk(step.content_delta)
                        elif (
                            not output_chunks
                            and step.is_complete_response
                            and step.content
                        ):
                            handled_special = True
                            output_chunks.append(step.content)
                            _touch("streaming")
                            if on_chunk is not None:
                                on_chunk(step.content)

                        if not handled_special:
                            _touch()
                except asyncio.CancelledError:
                    aborted_intentionally = True
                    status = "cancelled"
                    error_detail = "Task was cancelled before completion."
                    with contextlib.suppress(BaseException):
                        await _force_abort_agent(agent)
        except asyncio.CancelledError:
            aborted_intentionally = True
            status = "cancelled"
            error_detail = "Task was cancelled before completion."
            if active_agent is not None:
                with contextlib.suppress(BaseException):
                    await _force_abort_agent(active_agent)
        except Exception as exc:  # noqa: BLE001
            if not aborted_intentionally:
                status = "error"
                error_detail = f"{type(exc).__name__}: {exc}"
        finally:
            if run_id:
                self._active_agents.pop(run_id, None)
            if status == "completed" and tool_calls_executed == 0 and seen_active_tool_ids:
                tool_calls_executed = len(seen_active_tool_ids)
            if telemetry is not None:
                telemetry["thinking_chunks"] = thinking_chunks
                telemetry["tool_calls_executed"] = tool_calls_executed
                telemetry["tool_errors"] = tool_errors
                telemetry["last_tool_name"] = last_tool_name
                telemetry["turns"] = turns
                telemetry["phase"] = status if status != "completed" else "completed"
            partial_output = "".join(output_chunks).strip()
        return status, error_detail, partial_output

    def force_kill_active_processes(self) -> None:
        """Synchronously force-kill any active localharness child processes."""
        for agent in list(self._active_agents.values()):
            _, proc = _extract_harness_process(agent)
            _force_kill_process(proc)

    async def shutdown(self) -> None:
        """Cancel all running background jobs and terminate active localharness processes."""
        pending = [task for task in self._job_tasks.values() if not task.done()]
        for task in pending:
            task.cancel()
        for agent in list(self._active_agents.values()):
            with contextlib.suppress(BaseException):
                await _force_abort_agent(agent)
        if pending:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=3.0,
                )
        self.force_kill_active_processes()

    async def _run_delegation(
        self,
        spec: str,
        workspace_path: str,
        resolved_model: str,
        resolved_base_url: str,
        job: dict[str, Any] | None = None,
        *,
        idle_timeout_seconds: float | None = None,
        max_wall_seconds: float | None = None,
        disable_thinking: bool = False,
        is_async_job: bool = False,
    ) -> str:
        effective_disable_thinking = disable_thinking or _env_truthy(
            "OLLAMA_WORKER_DISABLE_THINKING"
        )
        effective_spec = spec
        config_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "base_url": resolved_base_url,
            "workspaces": [workspace_path],
            "policies": build_worker_policies(),
        }
        if effective_disable_thinking:
            config_kwargs["system_instructions"] = (
                "Execute tool calls directly and keep responses concise without verbose "
                "internal reasoning."
            )
            if "qwen3" in resolved_model.lower() and not spec.lstrip().startswith("/no_think"):
                effective_spec = f"/no_think\n{spec}"

        config = LocalOpenAIAgentConfig(**config_kwargs).lightweight()

        default_idle = _parse_positive_float_env(
            "OLLAMA_WORKER_IDLE_TIMEOUT_SECONDS", DEFAULT_IDLE_TIMEOUT_SECONDS
        )
        default_wall = (
            _parse_positive_float_env(
                "OLLAMA_WORKER_MAX_WALL_SECONDS", DEFAULT_ASYNC_MAX_WALL_SECONDS
            )
            if is_async_job
            else _parse_positive_float_env(
                "OLLAMA_WORKER_SYNC_MAX_WALL_SECONDS", DEFAULT_SYNC_MAX_WALL_SECONDS
            )
        )
        effective_idle = (
            float(idle_timeout_seconds)
            if idle_timeout_seconds is not None and idle_timeout_seconds > 0
            else default_idle
        )
        effective_wall = (
            float(max_wall_seconds)
            if max_wall_seconds is not None and max_wall_seconds > 0
            else default_wall
        )

        runtime_warnings: list[str] = []
        started = time.monotonic()
        telemetry: dict[str, Any] = (
            job
            if job is not None
            else {
                "phase": "queued",
                "chunks_received": 0,
                "thinking_chunks": 0,
                "tool_calls_executed": 0,
                "tool_errors": 0,
                "last_tool_name": "",
                "turns": 0,
                "last_activity_at": started,
                "elapsed_seconds": 0.0,
                "partial_output": "",
            }
        )
        run_id = job["job_id"] if job is not None else f"sync-{uuid.uuid4().hex[:8]}"

        def _on_chunk(token: str) -> None:
            now_ts = time.monotonic()
            telemetry["chunks_received"] = int(telemetry.get("chunks_received", 0)) + 1
            telemetry["partial_output"] = (
                str(telemetry.get("partial_output", "")) + token
            )[-1000:]
            telemetry["elapsed_seconds"] = round(now_ts - started, 2)
            telemetry["last_activity_at"] = now_ts

        try:
            async with self.slot_gate.acquire(workspace_path=workspace_path) as lease:
                if job is not None:
                    job["status"] = "running"
                    job["phase"] = "starting"
                    job["last_activity_at"] = time.monotonic()
                if lease.workspace_collision:
                    runtime_warnings.append(
                        "[WARNING] Another concurrent ollama-worker task is active in the same "
                        "workspace; use disjoint file scopes or separate git worktrees to avoid "
                        "overlapping edits."
                    )
                if (
                    lease.active_slots > 1
                    and self.scaling_profile.effective_memory_gb <= 18.0
                    and any(
                        marker in resolved_model.lower()
                        for marker in ("gemma4", "14b", "32b", "36b")
                    )
                ):
                    runtime_warnings.append(
                        f"[WARNING] {lease.active_slots} concurrent workers detected on a "
                        f"{self.scaling_profile.effective_memory_gb:.1f} GB host using {resolved_model}; "
                        f"prefer {self.scaling_profile.recommended_parallel_model} for parallel "
                        "subagents to stay 100% GPU without swap/KV-cache spill."
                    )

                exec_sig = inspect.signature(self._execute_agent)
                supports_rich_kwargs = "telemetry" in exec_sig.parameters or any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in exec_sig.parameters.values()
                )
                if supports_rich_kwargs:
                    status, error_detail, partial_output = await self._execute_agent(
                        config,
                        effective_spec,
                        on_chunk=_on_chunk,
                        telemetry=telemetry,
                        idle_timeout_seconds=effective_idle,
                        max_wall_seconds=effective_wall,
                        run_id=run_id,
                    )
                else:
                    status, error_detail, partial_output = await self._execute_agent(
                        config,
                        effective_spec,
                        on_chunk=_on_chunk if job is not None else None,
                    )
        except asyncio.CancelledError:
            status = "cancelled"
            error_detail = "Task was cancelled before completion."
            partial_output = str(telemetry.get("partial_output", "")).strip()

        warning_block = ("\n" + "\n".join(runtime_warnings) + "\n") if runtime_warnings else ""
        tool_calls_executed = int(telemetry.get("tool_calls_executed", 0))

        if status == "completed":
            if _looks_like_raw_tool_call(partial_output) or (
                tool_calls_executed == 0 and _contains_raw_tool_call(partial_output)
            ):
                status = "tool_call_error"
                error_detail = (
                    f"Model {resolved_model!r} emitted a raw JSON tool call as plain text instead "
                    f"of structured tool_calls (tool_calls={tool_calls_executed}). Retry with a "
                    "reliable native tool-calling model (e.g. 'gemma4-32k:latest' or 'qwen3:8b')."
                )
            elif tool_calls_executed == 0:
                leaked_token = _find_leaked_control_token(partial_output)
                if leaked_token is not None:
                    status = "tool_call_error"
                    error_detail = (
                        f"Model {resolved_model!r} leaked raw template control token "
                        f"{leaked_token!r} with 0 executed tools; template/tool-call parser "
                        "stopped before executing any edits."
                    )

        if status == "completed":
            body = partial_output or "(No text output returned by local worker.)"
            rendered = (
                f"[ollama-worker | model={resolved_model} | workspace={workspace_path} | "
                f"tool_calls={tool_calls_executed}]\n"
                f"{warning_block}\n"
                f"{body}\n\n"
                "---\n"
                "Next step: Inspect `git diff` and `git status` in the workspace to verify "
                "all local changes before accepting."
            )
        else:
            lines = [
                f"[ollama-worker | status={status} | model={resolved_model} | "
                f"workspace={workspace_path} | tool_calls={tool_calls_executed}]",
            ]
            if runtime_warnings:
                lines.extend(runtime_warnings)
            lines.append(f"Diagnostic: {error_detail}")
            if partial_output:
                lines.append(f"\nPartial output before {status}:\n{partial_output}")
            lines.append(
                "\nNext step: Inspect `git diff` / `git status` if the worker made partial edits."
            )
            rendered = "\n".join(lines)

        if job is not None:
            job["status"] = status
            job["phase"] = status
            job["elapsed_seconds"] = round(time.monotonic() - started, 2)
            job["result"] = rendered

        return rendered

    async def delegate_coding_task(
        self,
        spec: str,
        workspace_path: str,
        model: str = "",
        base_url: str = "",
        idle_timeout_seconds: int = 0,
        max_wall_seconds: int = 0,
        disable_thinking: bool = False,
    ) -> str:
        """Delegate a self-contained Radical Spec coding task to a local Ollama model.

        Args:
        spec: Complete, self-contained Radical Spec (objective, file scope, contracts, ordered steps, and verification command).
        workspace_path: Absolute path to the target project workspace directory.
        model: Local Ollama model tag (e.g. 'gemma4-32k:latest', 'qwen3:8b'). Leave empty to use provider default.
        base_url: Local OpenAI-compatible Ollama endpoint URL. Leave empty to use provider default.
        idle_timeout_seconds: Max seconds without step/chunk activity before aborting (0 = default).
        max_wall_seconds: Max total seconds before aborting (0 = default).
        disable_thinking: Hint to disable verbose reasoning (/no_think on qwen3).
        """
        err = self._validate_delegation_args(spec, workspace_path)
        if err is not None:
            return err

        resolved_model = (model or "").strip() or self.default_model
        resolved_base_url = (base_url or "").strip() or self.default_base_url
        return await self._run_delegation(
            spec=spec,
            workspace_path=workspace_path,
            resolved_model=resolved_model,
            resolved_base_url=resolved_base_url,
            idle_timeout_seconds=float(idle_timeout_seconds) if idle_timeout_seconds > 0 else None,
            max_wall_seconds=float(max_wall_seconds) if max_wall_seconds > 0 else None,
            disable_thinking=bool(disable_thinking),
            is_async_job=False,
        )

    async def start_coding_task(
        self,
        spec: str,
        workspace_path: str,
        model: str = "",
        base_url: str = "",
        idle_timeout_seconds: int = 0,
        max_wall_seconds: int = 0,
        disable_thinking: bool = False,
    ) -> str:
        """Start an asynchronous coding task in the background and return a job_id immediately."""
        err = self._validate_delegation_args(spec, workspace_path)
        if err is not None:
            return err

        resolved_model = (model or "").strip() or self.default_model
        resolved_base_url = (base_url or "").strip() or self.default_base_url
        job_id = uuid.uuid4().hex[:10]
        now_ts = time.monotonic()
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "phase": "queued",
            "model": resolved_model,
            "workspace_path": workspace_path,
            "started_at": now_ts,
            "last_activity_at": now_ts,
            "elapsed_seconds": 0.0,
            "chunks_received": 0,
            "thinking_chunks": 0,
            "tool_calls_executed": 0,
            "tool_errors": 0,
            "last_tool_name": "",
            "turns": 0,
            "partial_output": "",
            "result": "",
        }
        self._jobs[job_id] = job

        async def _runner() -> None:
            await self._run_delegation(
                spec=spec,
                workspace_path=workspace_path,
                resolved_model=resolved_model,
                resolved_base_url=resolved_base_url,
                job=job,
                idle_timeout_seconds=(
                    float(idle_timeout_seconds) if idle_timeout_seconds > 0 else None
                ),
                max_wall_seconds=float(max_wall_seconds) if max_wall_seconds > 0 else None,
                disable_thinking=bool(disable_thinking),
                is_async_job=True,
            )

        self._job_tasks[job_id] = asyncio.create_task(_runner())
        return (
            f"[ollama-worker | job_id={job_id} | status=queued | model={resolved_model} | "
            f"workspace={workspace_path}]\n"
            f"Task started asynchronously. Poll `check_coding_task(job_id={job_id!r}, "
            "wait_seconds=30)` until completion, or `cancel_coding_task` to abort."
        )

    async def check_coding_task(self, job_id: str, wait_seconds: int = 0) -> str:
        """Check status or await completion (up to 60s per poll) of an async coding job."""
        cleaned_id = (job_id or "").strip()
        job = self._jobs.get(cleaned_id)
        if job is None:
            known = ", ".join(sorted(self._jobs.keys())) or "none"
            return f"Error: Unknown job_id {job_id!r}. Known jobs: {known}."

        task = self._job_tasks.get(cleaned_id)
        clamped_wait = max(0, min(60, int(wait_seconds)))
        if clamped_wait > 0 and task is not None and not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=float(clamped_wait))
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        now_ts = time.monotonic()
        elapsed = round(now_ts - float(job["started_at"]), 2)
        if job["status"] not in {"queued", "running", "cancelling"} and job["result"]:
            return (
                f"[ollama-worker | job_id={cleaned_id} | status={job['status']} | "
                f"elapsed={job['elapsed_seconds']}s]\n\n{job['result']}"
            )

        idle = round(max(0.0, now_ts - float(job.get("last_activity_at", job["started_at"]))), 1)
        last_tool = str(job.get("last_tool_name") or "")
        tool_suffix = f" | last_tool={last_tool}" if last_tool else ""
        tail = (job.get("partial_output") or "").strip()[-500:]
        lines = [
            f"[ollama-worker | job_id={cleaned_id} | status={job['status']} | "
            f"phase={job.get('phase', job['status'])} | model={job['model']} | "
            f"elapsed={elapsed}s | idle={idle}s | chunks={job['chunks_received']} | "
            f"thinking_chunks={job.get('thinking_chunks', 0)} | "
            f"tool_calls={job.get('tool_calls_executed', 0)} | "
            f"turns={job.get('turns', 0)}{tool_suffix}]",
            "Task is still in progress. Call `check_coding_task` again with `wait_seconds=30`.",
        ]
        if tail:
            lines.append(f"\nRecent output tail:\n{tail}")
        return "\n".join(lines)

    async def cancel_coding_task(self, job_id: str) -> str:
        """Cancel a queued or running asynchronous coding job."""
        cleaned_id = (job_id or "").strip()
        job = self._jobs.get(cleaned_id)
        if job is None:
            return f"Error: Unknown job_id {job_id!r}."

        task = self._job_tasks.get(cleaned_id)
        if task is None or task.done():
            return (
                f"[ollama-worker | job_id={cleaned_id} | status={job['status']}] "
                "Job already finished; nothing to cancel."
            )

        job["status"] = "cancelling"
        job["phase"] = "cancelling"
        task.cancel()

        active_agent = self._active_agents.get(cleaned_id)
        if active_agent is not None:
            with contextlib.suppress(BaseException):
                await _force_abort_agent(active_agent)

        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        if task.done():
            job["status"] = "cancelled"
            job["phase"] = "cancelled"
            return (
                f"[ollama-worker | job_id={cleaned_id} | status=cancelled] "
                "Job cancelled and worker slot released."
            )

        return (
            f"[ollama-worker | job_id={cleaned_id} | status=cancelling] "
            "Cancellation requested; worker slot is still unwinding. "
            "Poll `check_coding_task` to confirm release."
        )

    def models_snapshot_json(self) -> str:
        """Return installed Ollama models (or connection diagnostic) serialized as JSON."""
        try:
            payload = self.fetch_ollama_tags()
            data: dict[str, Any] = {
                "endpoint": self.ollama_tags_url,
                "default_model": self.default_model,
                "models": payload.get("models") or [],
                "host_profile": self.scaling_profile.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001
            data = {
                "endpoint": self.ollama_tags_url,
                "default_model": self.default_model,
                "error": f"{type(exc).__name__}: {exc}",
                "models": [],
                "host_profile": self.scaling_profile.to_dict(),
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
            core: OllamaWorkerCore | None = None,
        ) -> None:
            self.core = core or OllamaWorkerCore(
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
            name="get_worker_scaling_profile",
            description="Inspect detected host hardware, active concurrency limits, recommended models, and scaling warnings.",
        )
        def get_worker_scaling_profile(self) -> str:
            """Return detected host hardware profile, concurrency limits, and warnings."""
            return self.core.get_worker_scaling_profile()

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
            idle_timeout_seconds: int = 0,
            max_wall_seconds: int = 0,
            disable_thinking: bool = False,
        ) -> str:
            """Delegate a self-contained Radical Spec coding task to a local Ollama model.

            Args:
            spec: Complete, self-contained Radical Spec (objective, file scope, contracts, ordered steps, and verification command).
            workspace_path: Absolute path to the target project workspace directory.
            model: Local Ollama model tag (e.g. 'gemma4-32k:latest', 'qwen2.5-coder:7b'). Leave empty to use provider default.
            base_url: Local OpenAI-compatible Ollama endpoint URL. Leave empty to use provider default.
            idle_timeout_seconds: Max seconds without step/chunk activity before aborting (0 = default).
            max_wall_seconds: Max total seconds before aborting (0 = default).
            disable_thinking: Hint to disable verbose reasoning (/no_think on qwen3).
            """
            return await self.core.delegate_coding_task(
                spec=spec,
                workspace_path=workspace_path,
                model=model,
                base_url=base_url,
                idle_timeout_seconds=idle_timeout_seconds,
                max_wall_seconds=max_wall_seconds,
                disable_thinking=disable_thinking,
            )

        @tool(
            name="start_coding_task",
            description="Start an asynchronous coding task in the background and return a job_id immediately.",
        )
        async def start_coding_task(
            self,
            spec: str,
            workspace_path: str,
            model: str = "",
            base_url: str = "",
            idle_timeout_seconds: int = 0,
            max_wall_seconds: int = 0,
            disable_thinking: bool = False,
        ) -> str:
            """Start an asynchronous coding task in the background and return a job_id immediately."""
            return await self.core.start_coding_task(
                spec=spec,
                workspace_path=workspace_path,
                model=model,
                base_url=base_url,
                idle_timeout_seconds=idle_timeout_seconds,
                max_wall_seconds=max_wall_seconds,
                disable_thinking=disable_thinking,
            )

        @tool(
            name="check_coding_task",
            description="Check status or await completion (up to 60s per poll) of an async coding job.",
        )
        async def check_coding_task(
            self,
            job_id: str,
            wait_seconds: int = 0,
        ) -> str:
            """Check status or await completion (up to 60s per poll) of an async coding job."""
            return await self.core.check_coding_task(
                job_id=job_id,
                wait_seconds=wait_seconds,
            )

        @tool(
            name="cancel_coding_task",
            description="Cancel a queued or running asynchronous coding job.",
        )
        async def cancel_coding_task(self, job_id: str) -> str:
            """Cancel a queued or running asynchronous coding job."""
            return await self.core.cancel_coding_task(job_id=job_id)

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
