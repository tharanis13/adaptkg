"""Locate the project root and the local MetaQA dataset directory."""

from __future__ import annotations

from pathlib import Path


def _kb_exists(root: Path) -> bool:
    return (root / "metaQA" / "kb.txt").is_file() or (root / "MetaQA" / "kb.txt").is_file()


def find_project_root() -> Path:
    """Find sem7pro by walking up until MetaQA/kb.txt is present."""
    search_from = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    seen: set[Path] = set()
    for start in search_from:
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if _kb_exists(candidate):
                return candidate
    raise FileNotFoundError(
        "Could not locate the project root containing MetaQA/kb.txt."
    )


def find_metaqa_dir(project_root: Path | None = None) -> Path:
    """Locate MetaQA regardless of folder capitalization."""
    root = project_root or find_project_root()
    candidates = [
        root / "metaQA",
        root / "MetaQA",
        root / "METAQA",
    ]
    for path in candidates:
        if path.is_dir() and (path / "kb.txt").is_file():
            return path

    matches = [
        p
        for p in root.iterdir()
        if p.is_dir() and p.name.lower() == "metaqa" and (p / "kb.txt").is_file()
    ]
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find MetaQA/kb.txt under project root: {root}")


def find_kb_path(project_root: Path | None = None) -> Path:
    return find_metaqa_dir(project_root) / "kb.txt"
