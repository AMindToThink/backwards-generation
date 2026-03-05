# Conversation Logs

Archived Claude Code conversation logs for the backwards-generation project.

| # | File | Description |
|---|------|-------------|
| 1 | `01_initial_implementation` | Initial project implementation — backward sampling with GPT-2 |
| 2 | `02_probability_math_audit` | Model upgrade plan and probability math audit |
| 3 | `03_forward_prompt_heuristic` | Forward prompt heuristic for candidate ordering |
| 4 | `04_reversed_sequence_heuristic` | Reversed-sequence heuristic for candidate ordering |
| 5 | `05_sandbox_test` | Brief sandbox/bash testing session |
| 6 | `06_archive_logs` | This archival session |

Each session has a `.jsonl` (raw log) and `.md` (readable markdown) file.

To regenerate the markdown files:
```bash
uv run python scripts/jsonl_to_markdown.py
```
