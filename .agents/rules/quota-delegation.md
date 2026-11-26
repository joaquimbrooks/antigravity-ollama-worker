---
trigger: always_on
description: Enforces cloud-architect / local-executor delegation via the radical-spec-delegate skill to preserve cloud quota.
---

Default to the `radical-spec-delegate` skill and the `ollama-worker` MCP server (`delegate_coding_task`) for code implementation, refactoring, boilerplate, test generation, and iterative test-fix debugging loops to preserve monthly cloud quota.

Keep cloud turns focused on planning, minimal signature inspection, writing self-contained Radical Specs, and reviewing `git diff` outputs. Only perform edits directly in the cloud for tiny (<10 lines in 1 file) changes or pure Q&A.
