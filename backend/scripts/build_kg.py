"""Load the local MetaQA knowledge graph and print statistics.

Run from the project root:
    python backend/scripts/build_kg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.services.knowledge_graph import (
    find_matching_entities,
    get_graph_statistics,
    load_graph,
    normalize_entity_name,
)
from backend.utils.paths import find_kb_path, find_project_root


def main() -> None:
    project_root = find_project_root()
    kb_path = find_kb_path(project_root)
    kg = load_graph(kb_path)
    stats = get_graph_statistics(kg)

    print("KNOWLEDGE GRAPH REPORT")
    print("----------------------")
    print(f"Project root: {project_root}")
    print(f"KB file:      {kb_path}")
    print()
    print(f"KG triples / edges: {stats['num_triples']}")
    print(f"Unique entities / nodes: {stats['num_entities']}")
    print(f"Unique relation types: {stats['num_relations']}")
    print(f"Malformed KB lines skipped: {stats['malformed_lines']}")
    print(f"Normalized lookup keys: {stats['index_size']}")
    print()
    print("Relation types:")
    for rel, count in sorted(stats["relation_counts"].items()):
        print(f"  - {rel} ({count})")
    print()
    print("LOOKUP DEMOS (names taken from kb.txt, not QA files)")
    print("----------------------------------------------------")
    demos = [
        "Kismet",
        "kismet",
        "  WILLIAM   DIETERLE ",
        "[Marlene Dietrich]",
        "this-entity-does-not-exist",
    ]
    for mention in demos:
        matches = find_matching_entities(kg, mention)
        print(f"  {mention!r} -> {matches}  (normalized={normalize_entity_name(mention)!r})")


if __name__ == "__main__":
    main()
