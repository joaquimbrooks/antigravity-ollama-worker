"""Safety policies for autonomous local Ollama coding workers."""

from __future__ import annotations

from typing import Any

from google.antigravity import policy

BLOCKED_COMMAND_PATTERNS: tuple[str, ...] = (
    "rm -rf",
    "sudo ",
    "git push",
    "git reset --hard",
    "curl ",
    "wget ",
    "mkfs",
    "dd if=",
)


def is_blocked_command(args: dict[str, Any]) -> bool:
    """Return True if the run_command CommandLine matches a blocked pattern."""
    command_line = str(args.get("CommandLine", "") or "").lower()
    return any(pattern in command_line for pattern in BLOCKED_COMMAND_PATTERNS)


def build_worker_policies() -> list[policy.Policy]:
    """Build the autonomous local worker policies with destructive-command guard."""
    return [
        policy.allow_all(),
        policy.deny(
            "run_command",
            when=is_blocked_command,
            name="block_destructive_commands",
            reason=(
                "Blocked destructive or network-exfiltration command pattern "
                "by ollama-worker safety policy."
            ),
        ),
    ]


__all__ = [
    "BLOCKED_COMMAND_PATTERNS",
    "is_blocked_command",
    "build_worker_policies",
]
