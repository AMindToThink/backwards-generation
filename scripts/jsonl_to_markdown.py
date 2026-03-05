#!/usr/bin/env python3
"""Convert Claude Code JSONL conversation logs to readable Markdown."""

import json
import re
import sys
from pathlib import Path


def clean_system_reminders(text: str) -> str:
    """Remove <system-reminder> tags and their contents."""
    return re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL).strip()


def render_content_blocks(blocks: list | str) -> str:
    """Render message content blocks to markdown."""
    if isinstance(blocks, str):
        cleaned = clean_system_reminders(blocks)
        return cleaned if cleaned else ""

    parts: list[str] = []
    has_non_tool_result = False

    for block in blocks:
        if isinstance(block, str):
            cleaned = clean_system_reminders(block)
            if cleaned:
                parts.append(cleaned)
                has_non_tool_result = True
            continue

        if not isinstance(block, dict):
            continue

        btype = block.get("type", "")

        if btype == "text":
            cleaned = clean_system_reminders(block.get("text", ""))
            if cleaned:
                parts.append(cleaned)
                has_non_tool_result = True

        elif btype == "tool_use":
            tool_name = block.get("name", "unknown")
            tool_input = block.get("input", {})
            # Show concise tool input
            if isinstance(tool_input, dict):
                summary_parts = []
                for k, v in tool_input.items():
                    sv = str(v)
                    if len(sv) > 200:
                        sv = sv[:200] + "..."
                    summary_parts.append(f"{k}: {sv}")
                input_summary = ", ".join(summary_parts)
            else:
                input_summary = str(tool_input)[:300]
            parts.append(f"**Tool: {tool_name}**\n```\n{input_summary}\n```")
            has_non_tool_result = True

        elif btype == "tool_result":
            # Abbreviate tool results
            content = block.get("content", "")
            if isinstance(content, list):
                text_parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                content = "\n".join(text_parts)
            if isinstance(content, str) and content.strip():
                abbreviated = content[:500]
                if len(content) > 500:
                    abbreviated += "\n... (truncated)"
                parts.append(f"<details><summary>Tool result</summary>\n\n```\n{abbreviated}\n```\n</details>")

        elif btype == "thinking":
            # Omit thinking blocks
            continue

    # Skip messages that are only tool_result
    if not has_non_tool_result:
        return ""

    return "\n\n".join(parts)


def convert_jsonl_to_markdown(jsonl_path: Path) -> str:
    """Convert a JSONL file to markdown."""
    lines_out: list[str] = []
    lines_out.append(f"# Conversation: {jsonl_path.stem}\n")

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type", "")
            if msg_type not in ("user", "assistant"):
                continue

            message = obj.get("message", {})
            content = message.get("content", "")

            rendered = render_content_blocks(content)
            if not rendered:
                continue

            role = "User" if msg_type == "user" else "Assistant"
            lines_out.append(f"## {role}\n")
            lines_out.append(rendered)
            lines_out.append("")

    return "\n".join(lines_out)


def main() -> None:
    if len(sys.argv) < 2:
        log_dir = Path("logs/conversation")
    else:
        log_dir = Path(sys.argv[1])

    jsonl_files = sorted(log_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {log_dir}")
        sys.exit(1)

    for jsonl_path in jsonl_files:
        md_path = jsonl_path.with_suffix(".md")
        print(f"Converting {jsonl_path.name} -> {md_path.name}")
        markdown = convert_jsonl_to_markdown(jsonl_path)
        md_path.write_text(markdown)

    print(f"Done. Converted {len(jsonl_files)} files.")


if __name__ == "__main__":
    main()
