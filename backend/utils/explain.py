"""Turn stored KG traversal steps into readable evidence-path text."""

from __future__ import annotations

from typing import Any


def format_evidence_path(path: list[dict[str, Any]] | None) -> str:
    """Render a real traversal path. Reverse hops keep original triples but walk tail->head."""
    if not path:
        return ""
    lines: list[str] = []
    previous: str | None = None
    for step in path:
        relation = str(step.get("relation", ""))
        if step.get("direction") == "reverse":
            src = str(step.get("object", ""))
            dst = str(step.get("subject", ""))
        else:
            src = str(step.get("subject", ""))
            dst = str(step.get("object", ""))
        if previous is None:
            lines.append(src)
        elif src != previous:
            lines.append(src)
        lines.append(f"↓ {relation}")
        lines.append(dst)
        previous = dst
    return "\n".join(lines)
