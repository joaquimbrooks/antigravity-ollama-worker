"""Standalone smoke test for the mcp-harness-backed ollama-worker server."""

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
from mcp_harness.providers.base import BaseProvider

# Ensure local imports work regardless of current working directory.
WORKER_DIR = Path(__file__).resolve().parent
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

import server  # noqa: E402


async def verify_provider_and_protocol() -> bool:
    """Verify BaseProvider inheritance, JSON-RPC 2.0 routing, schemas, resources, and prompts."""
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
    if "list_local_ollama_models" not in tools or "delegate_coding_task" not in tools:
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
    json.loads(contents[0].get("text", ""))

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
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
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
        help="Only verify protocol, STDIO wire purity, and deny policy without requiring Ollama.",
    )
    args = parser.parse_args()

    print(
        f"[PASS] Imported server.py cleanly "
        f"(MCPHarness server_name={server.harness.server_name!r})."
    )

    protocol_ok = await verify_provider_and_protocol()
    if not protocol_ok:
        return 1

    stdio_ok = verify_stdio_wire_purity()
    if not stdio_ok:
        return 1

    policy_ok = await verify_deny_policy()
    if not policy_ok:
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
