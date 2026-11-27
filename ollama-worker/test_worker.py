"""Standalone smoke test for ollama-worker (supports both mcp-harness and MCPServer fallback)."""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from google.antigravity import Agent, LocalOpenAIAgentConfig, types
from google.antigravity.hooks import hooks, policy

# Ensure local imports work regardless of current working directory.
WORKER_DIR = Path(__file__).resolve().parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

import server  # noqa: E402


async def verify_mcpserver_fallback() -> bool:
    """Verify the zero-extra-dependency MCPServer fallback registers tools, resource, and prompt."""
    mcp_srv = server.create_mcpserver_fallback()

    tools = {t.name: t for t in await mcp_srv.list_tools()}
    expected_tools = {
        "list_local_ollama_models",
        "get_worker_scaling_profile",
        "delegate_coding_task",
        "start_coding_task",
        "check_coding_task",
        "cancel_coding_task",
    }
    if not expected_tools.issubset(tools.keys()):
        print(f"[FAIL] Missing expected tools in MCPServer fallback: {list(tools.keys())}")
        return False

    resources = {str(r.uri): r for r in await mcp_srv.list_resources()}
    if "ollama://models" not in resources:
        print(f"[FAIL] Missing ollama://models in MCPServer fallback: {list(resources.keys())}")
        return False

    res_contents = await mcp_srv.read_resource("ollama://models")
    if not res_contents:
        print("[FAIL] Empty resource response from MCPServer fallback.")
        return False
    snapshot = json.loads(res_contents[0].content)
    if "host_profile" not in snapshot:
        print(f"[FAIL] Missing 'host_profile' in ollama://models snapshot: {list(snapshot.keys())}")
        return False

    prompts = {p.name: p for p in await mcp_srv.list_prompts()}
    if "radical_spec" not in prompts:
        print(f"[FAIL] Missing radical_spec in MCPServer fallback: {list(prompts.keys())}")
        return False

    prompt_res = await mcp_srv.get_prompt(
        "radical_spec",
        {"objective": "Add unit tests", "workspace_path": "/tmp/example"},
    )
    if not prompt_res.messages or "Add unit tests" not in prompt_res.messages[0].content.text:
        print(f"[FAIL] Unexpected MCPServer prompt response: {prompt_res}")
        return False

    tool_res = await mcp_srv.call_tool(
        "delegate_coding_task",
        {"spec": "Test spec", "workspace_path": "relative/path"},
    )
    # mcp 2.x returns CallToolResult(.content); older versions return sequence/tuple
    content_items = getattr(tool_res, "content", None)
    if content_items is None:
        content_items = tool_res[0] if isinstance(tool_res, tuple) else tool_res
    text = getattr(content_items[0], "text", str(content_items[0])) if content_items else ""
    if "must be an absolute directory path" not in text:
        print(f"[FAIL] Unexpected MCPServer fallback validation error: {tool_res!r}")
        return False

    print("[PASS] MCPServer fallback tools, resource, prompt, and validation verified.")
    return True


async def verify_provider_and_protocol() -> bool:
    """Verify optional mcp-harness BaseProvider inheritance and ProtocolEngine routing."""
    if not server.HAS_MCP_HARNESS:
        print("[SKIP] mcp-harness not installed; skipping BaseProvider/ProtocolEngine checks.")
        return True

    from mcp_harness.providers.base import BaseProvider

    provider = server.OllamaWorkerProvider()
    if not isinstance(provider, BaseProvider):
        print(f"[FAIL] OllamaWorkerProvider is not a BaseProvider: {type(provider)}")
        return False

    harness = server.create_harness()
    engine = harness.engine

    # 1. initialize
    init_raw = await engine.handle_message(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    )
    init_resp = json.loads(init_raw or "{}")
    server_info = init_resp.get("result", {}).get("serverInfo", {})
    if server_info.get("name") != "ollama-worker":
        print(f"[FAIL] Unexpected initialize response: {init_resp}")
        return False

    # 2. tools/list
    tools_raw = await engine.handle_message(
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    )
    tools_resp = json.loads(tools_raw or "{}")
    tools = {t["name"]: t for t in tools_resp.get("result", {}).get("tools", [])}
    expected_tools = {
        "list_local_ollama_models",
        "get_worker_scaling_profile",
        "delegate_coding_task",
        "start_coding_task",
        "check_coding_task",
        "cancel_coding_task",
    }
    if not expected_tools.issubset(tools.keys()):
        print(f"[FAIL] Missing expected tools in tools/list: {list(tools.keys())}")
        return False

    delegate_schema = tools["delegate_coding_task"].get("inputSchema", {})
    required_fields = set(delegate_schema.get("required", []))
    if not {"spec", "workspace_path"}.issubset(required_fields):
        print(f"[FAIL] delegate_coding_task required fields mismatch: {delegate_schema}")
        return False

    # 3. resources/list & resources/read
    res_list_raw = await engine.handle_message(
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}})
    )
    res_list_resp = json.loads(res_list_raw or "{}")
    uris = [r.get("uri") for r in res_list_resp.get("result", {}).get("resources", [])]
    if "ollama://models" not in uris:
        print(f"[FAIL] Missing ollama://models in resources/list: {uris}")
        return False

    res_read_raw = await engine.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "resources/read",
                "params": {"uri": "ollama://models"},
            }
        )
    )
    res_read_resp = json.loads(res_read_raw or "{}")
    contents = res_read_resp.get("result", {}).get("contents", [])
    if not contents or contents[0].get("uri") != "ollama://models":
        print(f"[FAIL] Unexpected resources/read response: {res_read_resp}")
        return False
    snapshot = json.loads(contents[0].get("text", ""))
    if "host_profile" not in snapshot:
        print(f"[FAIL] Missing 'host_profile' in ollama://models snapshot: {list(snapshot.keys())}")
        return False

    # 4. prompts/list & prompts/get
    prompts_list_raw = await engine.handle_message(
        json.dumps({"jsonrpc": "2.0", "id": 5, "method": "prompts/list", "params": {}})
    )
    prompts_list_resp = json.loads(prompts_list_raw or "{}")
    prompt_names = [p.get("name") for p in prompts_list_resp.get("result", {}).get("prompts", [])]
    if "radical_spec" not in prompt_names:
        print(f"[FAIL] Missing radical_spec in prompts/list: {prompt_names}")
        return False

    prompt_get_raw = await engine.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "prompts/get",
                "params": {
                    "name": "radical_spec",
                    "arguments": {
                        "objective": "Add unit tests",
                        "workspace_path": "/tmp/example",
                    },
                },
            }
        )
    )
    prompt_get_resp = json.loads(prompt_get_raw or "{}")
    messages = prompt_get_resp.get("result", {}).get("messages", [])
    if not messages or "Add unit tests" not in messages[0]["content"]["text"]:
        print(f"[FAIL] Unexpected prompts/get response: {prompt_get_resp}")
        return False

    # 5. tools/call validation guard (invalid relative workspace_path)
    bad_call_raw = await engine.handle_message(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {
                    "name": "delegate_coding_task",
                    "arguments": {"spec": "Test spec", "workspace_path": "relative/path"},
                },
            }
        )
    )
    bad_call_resp = json.loads(bad_call_raw or "{}")
    text = bad_call_resp.get("result", {}).get("content", [{}])[0].get("text", "")
    if "must be an absolute directory path" not in text:
        print(f"[FAIL] Unexpected validation error for relative workspace_path: {bad_call_resp}")
        return False

    print("[PASS] BaseProvider, ProtocolEngine, tools, resource, and prompt contracts verified.")
    return True


def verify_stdio_wire_purity() -> bool:
    """Spawn server.py over STDIO and verify stdout emits only clean JSON-RPC lines."""
    requests = [
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "smoke-test", "version": "0.1.0"},
                },
            }
        ),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
    ]
    proc = subprocess.run(
        [sys.executable, str(WORKER_DIR / "server.py")],
        input="\n".join(requests) + "\n",
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    if proc.returncode != 0:
        print(f"[FAIL] STDIO subprocess exited with code {proc.returncode}: {proc.stderr}")
        return False

    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(stdout_lines) != 2:
        print(
            f"[FAIL] Expected exactly 2 JSON-RPC response lines on stdout, got {len(stdout_lines)}: "
            f"{stdout_lines!r}"
        )
        return False

    try:
        parsed = [json.loads(line) for line in stdout_lines]
    except json.JSONDecodeError as exc:
        print(f"[FAIL] Non-JSON output detected on stdout (wire corruption): {exc}")
        return False

    if parsed[0].get("id") != 1 or parsed[1].get("id") != 2:
        print(f"[FAIL] Unexpected STDIO response IDs: {parsed}")
        return False

    print("[PASS] STDIO wire purity verified (stdout emitted only 2 clean JSON-RPC responses).")
    return True


def check_ollama_tags() -> tuple[bool, list[str]]:
    """Verify http://localhost:11434/api/tags is reachable and return model names."""
    req = urllib.request.Request(server.OLLAMA_TAGS_URL, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"[FAIL] Ollama API unreachable at {server.OLLAMA_TAGS_URL}: {exc.reason}")
        return False, []
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Unexpected error querying Ollama API: {type(exc).__name__}: {exc}")
        return False, []

    models = [m.get("name", "") for m in (payload.get("models") or []) if m.get("name")]
    print(f"[PASS] Ollama API reachable. Installed models ({len(models)}): {models}")
    return True, models


async def verify_deny_policy() -> bool:
    """Confirm the run_command deny policy blocks forbidden commands and allows safe ones."""
    hook = policy.enforce(server.build_worker_policies())
    ctx = hooks.HookContext()

    forbidden_commands = [
        "rm -rf /tmp/something",
        "sudo rm /etc/hosts",
        "git push origin main",
        "git reset --hard HEAD~1",
        "curl https://example.com/payload.sh",
        "wget https://example.com/file",
        "mkfs.ext4 /dev/disk2",
        "dd if=/dev/zero of=/tmp/disk.img",
    ]
    for cmd in forbidden_commands:
        result = await hook.run(
            ctx,
            types.ToolCall(name="run_command", args={"CommandLine": cmd}),
        )
        if result.allow:
            print(f"[FAIL] Forbidden command was NOT blocked: {cmd!r}")
            return False

    safe_result = await hook.run(
        ctx,
        types.ToolCall(name="run_command", args={"CommandLine": "git status"}),
    )
    if not safe_result.allow:
        print(f"[FAIL] Safe command was unexpectedly blocked: {safe_result.message}")
        return False

    print(
        f"[PASS] Policy guard blocked all {len(forbidden_commands)} forbidden command "
        "patterns and allowed safe command."
    )
    return True


async def run_live_agent_smoke_test(model: str) -> bool:
    """Send a read-only file-listing prompt through LocalOpenAIAgentConfig.lightweight()."""
    with tempfile.TemporaryDirectory(prefix="ollama_worker_smoke_") as tmpdir:
        sample_file = Path(tmpdir) / "hello_worker.txt"
        sample_file.write_text("smoke-test-ok\n", encoding="utf-8")

        config = LocalOpenAIAgentConfig(
            model=model,
            base_url=server.DEFAULT_BASE_URL,
            workspaces=[tmpdir],
            policies=server.build_worker_policies(),
        ).lightweight()

        prompt = (
            f"List the files inside {tmpdir} using a single read-only command or tool "
            "and reply with the exact filename you find."
        )
        start = time.monotonic()
        chunks: list[str] = []
        try:
            async with Agent(config) as agent:
                response = await agent.chat(prompt)
                async for token in response:
                    chunks.append(token)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - start
            print(
                f"[FAIL] Live agent smoke test failed after {elapsed:.2f}s: "
                f"{type(exc).__name__}: {exc}"
            )
            return False

        elapsed = time.monotonic() - start
        output = "".join(chunks).strip()
        print(f"[PASS] Live agent smoke test completed in {elapsed:.2f}s (model={model}):")
        print(output)
        return True


async def verify_hardware_and_scaling() -> bool:
    """Verify host hardware scaling tiers, env overrides, warnings, and WorkerSlotGate."""
    import hardware

    # 1. Synthetic hardware tier checks
    hw_16gb = hardware.HostHardware(
        os_name="Darwin",
        arch="arm64",
        cpu_brand="Apple M1",
        logical_cores=8,
        perf_cores=4,
        total_ram_gb=16.0,
        gpu_vram_gb=None,
        is_unified_memory=True,
    )
    prof_16gb = hardware.build_scaling_profile(hardware=hw_16gb, env={})
    if (
        prof_16gb.tier != "compact-<=16gb"
        or prof_16gb.max_parallel_workers != 1
        or prof_16gb.recommended_context_length != 16384
        or prof_16gb.recommended_single_model != "gemma4-32k:latest"
        or prof_16gb.recommended_parallel_model != "qwen3:4b"
    ):
        print(f"[FAIL] Unexpected compact 16GB scaling profile: {prof_16gb}")
        return False

    prof_16gb_32k = hardware.build_scaling_profile(
        hardware=hw_16gb, env={"OLLAMA_WORKER_CONTEXT_LENGTH": "32768"}
    )
    if prof_16gb_32k.recommended_context_length != 32768 or not any(
        "16384" in w for w in prof_16gb_32k.warnings
    ):
        print(f"[FAIL] Expected 16GB 32K context pressure warning: {prof_16gb_32k}")
        return False

    prof_16gb_parallel = hardware.build_scaling_profile(
        hardware=hw_16gb, env={"OLLAMA_WORKER_MAX_PARALLEL": "2"}
    )
    if (
        prof_16gb_parallel.max_parallel_workers != 2
        or not any("KV cache" in w for w in prof_16gb_parallel.warnings)
    ):
        print(f"[FAIL] Expected KV cache warning for 16GB parallel=2: {prof_16gb_parallel}")
        return False

    prof_qwen25 = hardware.build_scaling_profile(
        hardware=hw_16gb, env={"OLLAMA_WORKER_MODEL": "qwen2.5-coder:7b"}
    )
    if not any("raw JSON text" in w for w in prof_qwen25.warnings):
        print(f"[FAIL] Expected raw JSON tool_calls warning for qwen2.5-coder: {prof_qwen25}")
        return False

    hw_1core = hardware.HostHardware(
        os_name="Linux",
        arch="x86_64",
        cpu_brand="Generic CPU",
        logical_cores=1,
        perf_cores=1,
        total_ram_gb=16.0,
        gpu_vram_gb=None,
        is_unified_memory=False,
    )
    prof_1core = hardware.build_scaling_profile(hardware=hw_1core, env={})
    if prof_1core.max_parallel_workers != 1:
        print(f"[FAIL] Expected 1-core host max_parallel_workers=1, got {prof_1core.max_parallel_workers}")
        return False

    hw_32gb = hardware.HostHardware(
        os_name="Darwin",
        arch="arm64",
        cpu_brand="Apple M3 Pro",
        logical_cores=12,
        perf_cores=6,
        total_ram_gb=32.0,
        gpu_vram_gb=None,
        is_unified_memory=True,
    )
    prof_32gb = hardware.build_scaling_profile(hardware=hw_32gb, env={})
    if (
        prof_32gb.tier != "mid-24gb-36gb"
        or prof_32gb.max_parallel_workers != 2
        or prof_32gb.recommended_ollama_max_loaded_models != 2
    ):
        print(f"[FAIL] Unexpected mid 32GB scaling profile: {prof_32gb}")
        return False

    hw_64gb = hardware.HostHardware(
        os_name="Darwin",
        arch="arm64",
        cpu_brand="Apple M3 Max",
        logical_cores=16,
        perf_cores=12,
        total_ram_gb=64.0,
        gpu_vram_gb=None,
        is_unified_memory=True,
    )
    prof_64gb = hardware.build_scaling_profile(hardware=hw_64gb, env={})
    if (
        prof_64gb.tier != "high-48gb+"
        or prof_64gb.max_parallel_workers != 4
        or prof_64gb.recommended_context_length != 65536
    ):
        print(f"[FAIL] Unexpected high 64GB scaling profile: {prof_64gb}")
        return False

    # 2. Override checks & malformed input resilience
    prof_capped = hardware.build_scaling_profile(
        hardware=hw_32gb, env={"OLLAMA_WORKER_MAX_CORES": "1"}
    )
    if prof_capped.max_parallel_workers != 1:
        print(f"[FAIL] OLLAMA_WORKER_MAX_CORES=1 did not cap parallelism on 32GB host: {prof_capped}")
        return False

    # Raw JSON tool-call detector checks
    import provider

    if not provider._looks_like_raw_tool_call(
        '{"name": "read_files", "arguments": {"files": ["pyproject.toml"]}}'
    ):
        print("[FAIL] _looks_like_raw_tool_call missed plain JSON tool payload.")
        return False
    if not provider._looks_like_raw_tool_call(
        '```json\n{"name": "run_command", "arguments": {"CommandLine": "pytest"}}\n```'
    ):
        print("[FAIL] _looks_like_raw_tool_call missed fenced JSON tool payload.")
        return False
    if not provider._looks_like_raw_tool_call(
        '{"name": "replace_file_content", "parameters": {"TargetFile": "a.py"}}'
    ):
        print("[FAIL] _looks_like_raw_tool_call missed Llama 3.1 'parameters' payload.")
        return False
    if not provider._looks_like_raw_tool_call(
        '{"tool_calls": [{"function": {"name": "edit_file", "arguments": {}}}]}'
    ):
        print("[FAIL] _looks_like_raw_tool_call missed OpenAI 'tool_calls' wrapper.")
        return False
    if not provider._contains_raw_tool_call(
        'Here is the edit:\n```json\n{"name": "replace_file_content", "parameters": {"TargetFile": "filter.py", "ReplacementContent": "# ..."}}\n```'
    ):
        print("[FAIL] _contains_raw_tool_call missed prose + fenced 'parameters' block.")
        return False
    if provider._looks_like_raw_tool_call("Completed edits and all tests passed."):
        print("[FAIL] _looks_like_raw_tool_call falsely flagged normal prose.")
        return False

    prof_custom = hardware.build_scaling_profile(
        hardware=hw_16gb,
        env={
            "OLLAMA_WORKER_MAX_PARALLEL": "3",
            "OLLAMA_WORKER_MEMORY_GB": "16",
            "OLLAMA_WORKER_CONTEXT_LENGTH": "16384",
            "OLLAMA_WORKER_MODEL": "custom:7b",
            "OLLAMA_WORKER_PARALLEL_MODEL": "fast:3b",
            "OLLAMA_WORKER_MAX_CORES": "not-an-int",
        },
    )
    if (
        prof_custom.max_parallel_workers != 3
        or prof_custom.recommended_context_length != 16384
        or prof_custom.recommended_single_model != "custom:7b"
        or prof_custom.recommended_parallel_model != "fast:3b"
        or not any("oversubscription" in w.lower() for w in prof_custom.warnings)
        or not any("malformed" in w.lower() for w in prof_custom.warnings)
    ):
        print(f"[FAIL] Unexpected custom override / warning behavior: {prof_custom}")
        return False

    # 3. WorkerSlotGate concurrency + workspace collision checks
    with tempfile.TemporaryDirectory(prefix="ollama_slot_gate_") as lock_dir:
        gate = hardware.WorkerSlotGate(max_slots=2, lock_dir=lock_dir)
        ws_a = str(Path(lock_dir) / "ws_a")
        Path(ws_a).mkdir()

        async with gate.acquire(workspace_path=ws_a) as lease1:
            if lease1.workspace_collision or lease1.active_slots != 1:
                print(f"[FAIL] Unexpected first lease: {lease1}")
                return False

            acquired_third = asyncio.Event()

            async with gate.acquire(workspace_path=ws_a) as lease2:
                if not lease2.workspace_collision or lease2.active_slots != 2:
                    print(f"[FAIL] Expected workspace collision & 2 active slots on lease2: {lease2}")
                    return False

                async def _wait_third() -> None:
                    async with gate.acquire(workspace_path="") as lease3:
                        if lease3.workspace_collision:
                            raise AssertionError(f"Unexpected collision on lease3: {lease3}")
                        acquired_third.set()

                third_task = asyncio.create_task(_wait_third())
                await asyncio.sleep(0.15)
                if acquired_third.is_set():
                    print("[FAIL] Third task acquired slot before 2nd slot released.")
                    third_task.cancel()
                    return False

            await asyncio.wait_for(acquired_third.wait(), timeout=2.0)
            await third_task

    print("[PASS] Host hardware scaling profiles, overrides, warnings, and WorkerSlotGate verified.")
    return True


async def verify_async_job_lifecycle() -> bool:
    """Verify start_coding_task, check_coding_task, and cancel_coding_task with mocked agent execution."""
    import hardware
    import provider

    with tempfile.TemporaryDirectory(prefix="ollama_async_jobs_") as tmpdir:
        core = provider.OllamaWorkerCore()
        core.slot_gate = hardware.WorkerSlotGate(
            max_slots=1, lock_dir=Path(tmpdir) / "locks"
        )

        bad_start = await core.start_coding_task("Test spec", "relative/path")
        if "must be an absolute directory path" not in bad_start:
            print(f"[FAIL] Unexpected start_coding_task validation output: {bad_start}")
            return False

        gate_started = asyncio.Event()
        release_agent = asyncio.Event()

        async def _fake_execute(config, spec, on_chunk=None):  # noqa: ANN001
            gate_started.set()
            if on_chunk is not None:
                on_chunk("working...")
            await release_agent.wait()
            return "completed", "", "All tests passed."

        core._execute_agent = _fake_execute  # type: ignore[method-assign]

        start_msg = await core.start_coding_task("Implement X", tmpdir, model="qwen3:8b")
        if "job_id=" not in start_msg or "status=queued" not in start_msg:
            print(f"[FAIL] Unexpected start_coding_task output: {start_msg}")
            return False
        job_id = start_msg.split("job_id=", 1)[1].split(" ", 1)[0]

        await asyncio.wait_for(gate_started.wait(), timeout=2.0)
        mid_check = await core.check_coding_task(job_id, wait_seconds=0)
        if "status=running" not in mid_check or "working..." not in mid_check:
            print(f"[FAIL] Unexpected mid-run check_coding_task output: {mid_check}")
            return False

        release_agent.set()
        done_check = await core.check_coding_task(job_id, wait_seconds=5)
        if "status=completed" not in done_check or "All tests passed." not in done_check:
            print(f"[FAIL] Unexpected completed check_coding_task output: {done_check}")
            return False

        # Test cancellation on a second blocked job
        gate_started.clear()
        release_agent.clear()
        start_msg_2 = await core.start_coding_task("Long task", tmpdir, model="qwen3:8b")
        job_id_2 = start_msg_2.split("job_id=", 1)[1].split(" ", 1)[0]
        await asyncio.wait_for(gate_started.wait(), timeout=2.0)
        cancel_msg = await core.cancel_coding_task(job_id_2)
        if "status=cancelled" not in cancel_msg:
            print(f"[FAIL] Unexpected cancel_coding_task output: {cancel_msg}")
            return False

    print("[PASS] Async job lifecycle (start, poll/wait, complete, cancel) verified.")
    return True


async def verify_worker_reliability_fixes() -> bool:
    """Verify Issues A-E fixes: raw-tool detection, step telemetry, force-abort, watchdogs, and clean exit."""
    import signal
    import hardware
    import provider

    with tempfile.TemporaryDirectory(prefix="ollama_reliability_") as tmpdir:
        core = provider.OllamaWorkerCore()
        core.slot_gate = hardware.WorkerSlotGate(
            max_slots=1, lock_dir=Path(tmpdir) / "locks"
        )

        # --- Issue A: _run_delegation tool_call_error vs completed with real tool calls ---
        orig_execute = core._execute_agent

        async def _llama_prose_json(
            config, spec, on_chunk=None, *, telemetry=None, **kwargs
        ):  # noqa: ANN001, ARG001
            if telemetry is not None:
                telemetry["tool_calls_executed"] = 0
            return (
                "completed",
                "",
                'I will update the file:\n```json\n{"name": "replace_file_content", "parameters": {"TargetFile": "filter.py"}}\n```',
            )

        core._execute_agent = _llama_prose_json  # type: ignore[method-assign]
        res_llama = await core._run_delegation(
            spec="Edit filter.py",
            workspace_path=tmpdir,
            resolved_model="llama3.1:8b-instruct-q8_0",
            resolved_base_url=server.DEFAULT_BASE_URL,
        )
        if "status=tool_call_error" not in res_llama:
            print(f"[FAIL] Expected tool_call_error for prose+parameters with 0 tools: {res_llama}")
            return False

        async def _leaked_channel_zero_tools(
            config, spec, on_chunk=None, *, telemetry=None, **kwargs
        ):  # noqa: ANN001, ARG001
            if telemetry is not None:
                telemetry["tool_calls_executed"] = 0
            return "completed", "", "Executing changes now.<channel|>"

        core._execute_agent = _leaked_channel_zero_tools  # type: ignore[method-assign]
        res_channel = await core._run_delegation(
            spec="Edit filter.py",
            workspace_path=tmpdir,
            resolved_model="gemma4-32k:latest",
            resolved_base_url=server.DEFAULT_BASE_URL,
        )
        if "status=tool_call_error" not in res_channel or "<channel|>" not in res_channel:
            print(f"[FAIL] Expected tool_call_error for leaked <channel|>: {res_channel}")
            return False

        async def _completed_with_tools(
            config, spec, on_chunk=None, *, telemetry=None, **kwargs
        ):  # noqa: ANN001, ARG001
            if telemetry is not None:
                telemetry["tool_calls_executed"] = 2
            return (
                "completed",
                "",
                'Updated filter.py. Example payload:\n```json\n{"name": "edit_file", "parameters": {"x": 1}}\n```',
            )

        core._execute_agent = _completed_with_tools  # type: ignore[method-assign]
        res_ok = await core._run_delegation(
            spec="Edit filter.py",
            workspace_path=tmpdir,
            resolved_model="qwen3:8b",
            resolved_base_url=server.DEFAULT_BASE_URL,
        )
        if "status=tool_call_error" in res_ok or "tool_calls=2" not in res_ok:
            print(f"[FAIL] False-positive tool_call_error when tool_calls=2: {res_ok}")
            return False

        # --- Issue D: disable_thinking propagates /no_think + system_instructions ---
        captured_specs: list[tuple[str, Any]] = []

        async def _capture_config(
            config, spec, on_chunk=None, *, telemetry=None, **kwargs
        ):  # noqa: ANN001, ARG001
            captured_specs.append((spec, config.system_instructions))
            if telemetry is not None:
                telemetry["tool_calls_executed"] = 1
            return "completed", "", "Done."

        core._execute_agent = _capture_config  # type: ignore[method-assign]
        await core.delegate_coding_task(
            spec="Fix bug",
            workspace_path=tmpdir,
            model="qwen3:8b",
            disable_thinking=True,
        )
        if (
            not captured_specs
            or not captured_specs[0][0].startswith("/no_think\n")
            or not captured_specs[0][1]
        ):
            print(f"[FAIL] disable_thinking did not propagate to spec/config: {captured_specs}")
            return False

        # Restore real _execute_agent for Agent step telemetry, abort, and idle watchdog tests
        core._execute_agent = orig_execute  # type: ignore[method-assign]

        # --- Issue B: Real-time step telemetry (thinking, tool_running, streaming) ---
        orig_agent_cls = provider.Agent
        tool_running_seen = asyncio.Event()
        allow_tool_done = asyncio.Event()

        class _MockConversation:
            def __init__(self, proc=None) -> None:
                self.cancel_called = False
                self._connection = type(
                    "_Conn", (), {"_process": proc, "_disconnecting": False}
                )()

            async def send(self, prompt: str) -> None:
                pass

            async def cancel(self) -> None:
                self.cancel_called = True

            async def receive_steps(self):
                yield types.Step(
                    id="m:0",
                    step_index=0,
                    trajectory_id="t1",
                    source=types.StepSource.MODEL,
                    type=types.StepType.THINKING,
                    status=types.StepStatus.ACTIVE,
                    thinking_delta="thinking step",
                )
                yield types.Step(
                    id="m:1",
                    step_index=1,
                    trajectory_id="t1",
                    source=types.StepSource.MODEL,
                    type=types.StepType.TOOL_CALL,
                    status=types.StepStatus.ACTIVE,
                    tool_calls=[types.ToolCall(name="edit_file", id="tc-1", args={})],
                )
                tool_running_seen.set()
                await allow_tool_done.wait()
                yield types.Step(
                    id="m:1",
                    step_index=1,
                    trajectory_id="t1",
                    source=types.StepSource.MODEL,
                    type=types.StepType.TOOL_CALL,
                    status=types.StepStatus.DONE,
                    tool_calls=[types.ToolCall(name="edit_file", id="tc-1", args={})],
                )
                yield types.Step(
                    id="m:2",
                    step_index=2,
                    trajectory_id="t1",
                    source=types.StepSource.MODEL,
                    target=types.StepTarget.USER,
                    type=types.StepType.TEXT_RESPONSE,
                    status=types.StepStatus.DONE,
                    content_delta="Edited file successfully.",
                    is_complete_response=True,
                )

        class _TelemetryMockAgent:
            def __init__(self, config) -> None:  # noqa: ANN001
                self._conversation = _MockConversation()

            @property
            def conversation(self):
                return self._conversation

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

        provider.Agent = _TelemetryMockAgent  # type: ignore[misc]
        try:
            start_b = await core.start_coding_task("Update filter", tmpdir, model="qwen3:8b")
            job_id_b = start_b.split("job_id=", 1)[1].split(" ", 1)[0]
            await asyncio.wait_for(tool_running_seen.wait(), timeout=2.0)
            mid_b = await core.check_coding_task(job_id_b, wait_seconds=0)
            if (
                "phase=tool_running" not in mid_b
                or "thinking_chunks=1" not in mid_b
                or "last_tool=edit_file" not in mid_b
                or "turns=2" not in mid_b
            ):
                print(f"[FAIL] Telemetry missing mid-run phase/fields: {mid_b}")
                return False
            allow_tool_done.set()
            done_b = await core.check_coding_task(job_id_b, wait_seconds=5)
            if "status=completed" not in done_b or "tool_calls=1" not in done_b:
                print(f"[FAIL] Telemetry completed output unexpected: {done_b}")
                return False
        finally:
            provider.Agent = orig_agent_cls  # type: ignore[misc]

        # --- Issue C: Force-abort child process and release slot gate on cancel ---
        dummy_proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stall_started = asyncio.Event()

        class _StallConversation:
            def __init__(self, proc) -> None:  # noqa: ANN001
                self.cancel_called = False
                self._connection = type(
                    "_Conn", (), {"_process": proc, "_disconnecting": False}
                )()

            async def send(self, prompt: str) -> None:
                pass

            async def cancel(self) -> None:
                self.cancel_called = True

            async def receive_steps(self):
                yield types.Step(
                    id="m:0",
                    step_index=0,
                    trajectory_id="t2",
                    source=types.StepSource.MODEL,
                    target=types.StepTarget.USER,
                    type=types.StepType.TEXT_RESPONSE,
                    status=types.StepStatus.ACTIVE,
                    content_delta="Executing changes now.<channel|>",
                )
                stall_started.set()
                await asyncio.sleep(30)

        stall_conv = _StallConversation(dummy_proc)

        class _StallMockAgent:
            def __init__(self, config) -> None:  # noqa: ANN001
                self._conversation = stall_conv

            @property
            def conversation(self):
                return self._conversation

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return False

        provider.Agent = _StallMockAgent  # type: ignore[misc]
        try:
            start_c = await core.start_coding_task("Stalled task", tmpdir, model="gemma4-32k:latest")
            job_id_c = start_c.split("job_id=", 1)[1].split(" ", 1)[0]
            await asyncio.wait_for(stall_started.wait(), timeout=2.0)
            cancel_c = await core.cancel_coding_task(job_id_c)
            if "status=cancelled" not in cancel_c:
                print(f"[FAIL] Expected status=cancelled on cancel: {cancel_c}")
                return False
            if not stall_conv.cancel_called or dummy_proc.poll() is None:
                print("[FAIL] Expected conversation.cancel() and terminated subprocess on cancel.")
                return False
            async with asyncio.timeout(1.0):
                async with core.slot_gate.acquire(workspace_path=tmpdir):
                    pass

            # --- Issue D: Idle timeout aborts stalled stream & surfaces leaked token ---
            dummy_proc_d = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stall_started.clear()
            stall_conv_d = _StallConversation(dummy_proc_d)

            class _StallMockAgentD(_StallMockAgent):
                def __init__(self, config) -> None:  # noqa: ANN001
                    self._conversation = stall_conv_d

            provider.Agent = _StallMockAgentD  # type: ignore[misc]
            timeout_res = await core.delegate_coding_task(
                spec="Stall test",
                workspace_path=tmpdir,
                model="gemma4-32k:latest",
                idle_timeout_seconds=1,
            )
            if (
                "status=timeout" not in timeout_res
                or "<channel|>" not in timeout_res
                or dummy_proc_d.poll() is None
            ):
                print(f"[FAIL] Idle timeout did not abort/diagnose properly: {timeout_res}")
                return False
        finally:
            provider.Agent = orig_agent_cls  # type: ignore[misc]
            if dummy_proc.poll() is None:
                dummy_proc.kill()
                dummy_proc.wait(timeout=1.0)
            if "dummy_proc_d" in locals() and dummy_proc_d.poll() is None:
                dummy_proc_d.kill()
                dummy_proc_d.wait(timeout=1.0)

        # --- Issue C (part 2): slow unwinding reports status=cancelling rather than false cancelled ---
        hold_unwind = asyncio.Event()
        unwinding_started = asyncio.Event()

        async def _slow_unwind(config, spec, on_chunk=None):  # noqa: ANN001
            unwinding_started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await hold_unwind.wait()
                return "cancelled", "cancelled", ""
            return "completed", "", ""

        core._execute_agent = _slow_unwind  # type: ignore[method-assign]
        start_slow = await core.start_coding_task("Slow unwind", tmpdir, model="qwen3:8b")
        job_id_slow = start_slow.split("job_id=", 1)[1].split(" ", 1)[0]
        await asyncio.wait_for(unwinding_started.wait(), timeout=2.0)
        cancel_slow = await core.cancel_coding_task(job_id_slow)
        hold_unwind.set()
        if "status=cancelling" not in cancel_slow:
            print(f"[FAIL] Expected status=cancelling while slot unwinds: {cancel_slow}")
            return False
        await core.check_coding_task(job_id_slow, wait_seconds=2)

    # --- Issue E: server.py exits cleanly on SIGTERM (with open stdin) and on stdin EOF ---
    init_req = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "sig-test", "version": "0.1.0"},
            },
        }
    )
    for mode_name in ("sigterm", "eof"):
        srv_proc = subprocess.Popen(
            [sys.executable, str(WORKER_DIR / "server.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert srv_proc.stdin is not None
            assert srv_proc.stdout is not None
            srv_proc.stdin.write(init_req + "\n")
            srv_proc.stdin.flush()
            init_line = srv_proc.stdout.readline()
            if not init_line or json.loads(init_line).get("id") != 1:
                print(f"[FAIL] server.py ({mode_name}) failed initialize handshake: {init_line!r}")
                return False
            t0 = time.monotonic()
            if mode_name == "sigterm":
                srv_proc.send_signal(signal.SIGTERM)
            else:
                srv_proc.stdin.close()
            rc = srv_proc.wait(timeout=2.0)
            elapsed_exit = time.monotonic() - t0
            if rc != 0 or elapsed_exit >= 2.0:
                print(f"[FAIL] server.py ({mode_name}) exited rc={rc} in {elapsed_exit:.2f}s")
                return False
        finally:
            if srv_proc.poll() is None:
                srv_proc.kill()
                srv_proc.wait(timeout=1.0)

    print("[PASS] Worker reliability fixes (Issues A-E) verified.")
    return True


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test ollama-worker.")
    parser.add_argument(
        "--model",
        default=None,
        help="Ollama model tag to use for the live smoke test.",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Only verify protocol, fallback, STDIO wire purity, and deny policy.",
    )
    args = parser.parse_args()

    mode = "mcp-harness" if server.HAS_MCP_HARNESS else "MCPServer-fallback"
    print(f"[PASS] Imported server.py cleanly (active runtime={mode}).")

    if not await verify_mcpserver_fallback():
        return 1

    if not await verify_provider_and_protocol():
        return 1

    if not verify_stdio_wire_purity():
        return 1

    if not await verify_deny_policy():
        return 1

    if not await verify_hardware_and_scaling():
        return 1

    if not await verify_async_job_lifecycle():
        return 1

    if not await verify_worker_reliability_fixes():
        return 1

    if args.offline_only:
        print("[SKIP] Live Ollama checks skipped (--offline-only).")
        return 0

    reachable, models = check_ollama_tags()
    if not reachable:
        print(
            "[WARN] Skipping live LocalOpenAIAgentConfig test because Ollama is not running. "
            "Start Ollama and re-run without --offline-only."
        )
        return 1

    if not models:
        print(
            "[WARN] Ollama is running but has 0 models pulled. "
            "Pull a model (e.g. `ollama pull qwen3:14b`) and re-run."
        )
        return 1

    chosen_model = args.model or (
        server.DEFAULT_MODEL if server.DEFAULT_MODEL in models else models[0]
    )
    live_ok = await run_live_agent_smoke_test(chosen_model)
    return 0 if live_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
