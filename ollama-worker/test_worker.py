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
    if (
        "list_local_ollama_models" not in tools
        or "delegate_coding_task" not in tools
        or "get_worker_scaling_profile" not in tools
    ):
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
    if (
        "list_local_ollama_models" not in tools
        or "delegate_coding_task" not in tools
        or "get_worker_scaling_profile" not in tools
    ):
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
        or prof_16gb.recommended_single_model != "gemma4-32k:latest"
        or prof_16gb.recommended_parallel_model != "qwen3:4b"
    ):
        print(f"[FAIL] Unexpected compact 16GB scaling profile: {prof_16gb}")
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
