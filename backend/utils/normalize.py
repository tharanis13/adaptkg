"""Entity-name normalization used only for lookup, never to rename graph nodes."""

from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")


def normalize_entity(text: str) -> str:
    """Trim, collapse whitespace, and lowercase an entity mention."""
    collapsed = _WHITESPACE.sub(" ", text.strip())
    return collapsed.casefold()


def normalize_entity_name(name: str) -> str:
    """Public alias required by Phase 2."""
    return normalize_entity(name)


def extract_mention(text: str) -> str:
    """Use bracketed mention if present; otherwise the full trimmed string."""
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]") and stripped.count("[") == 1:
        return stripped[1:-1].strip()
    return stripped
