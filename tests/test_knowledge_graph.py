"""Tests for the Phase 2 Knowledge Graph layer."""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.services.knowledge_graph import (
    KnowledgeGraph,
    find_matching_entities,
    get_graph_statistics,
    load_graph,
    normalize_entity_name,
)
from backend.utils.normalize import normalize_entity
from backend.utils.paths import find_kb_path, find_project_root

# Counts from Phase 1 inspection of the real kb.txt
EXPECTED_TRIPLES = 134741
EXPECTED_ENTITIES = 43234
EXPECTED_RELATIONS = 9


def test_normalize_entity_name_trims_and_lowercases() -> None:
    assert normalize_entity_name("  WILLIAM   DIETERLE ") == "william dieterle"
    assert normalize_entity("Kismet") == "kismet"


def test_multiedges_and_original_node_ids_preserved() -> None:
    kg = KnowledgeGraph.from_triples(
        [
            ("Kismet", "directed_by", "William Dieterle"),
            ("Kismet", "written_by", "William Dieterle"),
            ("Kismet", "starred_actors", "Marlene Dietrich"),
        ]
    )
    assert isinstance(kg.graph, nx.MultiDiGraph)
    assert kg.has_entity("Kismet")
    assert kg.has_entity("William Dieterle")
    assert "kismet" not in kg.graph

    relations = {
        data["relation"]
        for data in kg.graph.get_edge_data("Kismet", "William Dieterle").values()
    }
    assert relations == {"directed_by", "written_by"}
    assert kg.graph.number_of_edges() == 3


def test_lookup_is_exact_normalized_match_only() -> None:
    kg = KnowledgeGraph.from_triples(
        [("Kismet", "directed_by", "William Dieterle")]
    )
    assert kg.lookup_entity("kismet") == ["Kismet"]
    assert find_matching_entities(kg, "[  WILLIAM DIETERLE ]") == ["William Dieterle"]
    assert find_matching_entities(kg, "unknown person") == []


@pytest.fixture(scope="module")
def real_kg() -> KnowledgeGraph:
    kb_path = find_kb_path(find_project_root())
    if not kb_path.is_file():
        pytest.skip(f"kb.txt not found at {kb_path}")
    return load_graph(kb_path)


def test_real_graph_loads_and_matches_phase1_counts(real_kg: KnowledgeGraph) -> None:
    stats = get_graph_statistics(real_kg)
    assert stats["num_triples"] == EXPECTED_TRIPLES
    assert stats["num_nodes"] == EXPECTED_ENTITIES
    assert stats["num_entities"] == EXPECTED_ENTITIES
    assert stats["num_relations"] == EXPECTED_RELATIONS
    assert stats["malformed_lines"] == 0
    assert "directed_by" in stats["relation_counts"]
    assert sum(stats["relation_counts"].values()) == EXPECTED_TRIPLES


def test_real_entities_and_triples_from_kb(real_kg: KnowledgeGraph) -> None:
    """Use actual first-file triples from metaQA/kb.txt (Kismet)."""
    assert real_kg.has_entity("Kismet")
    assert real_kg.has_entity("William Dieterle")
    assert real_kg.has_entity("Marlene Dietrich")

    assert find_matching_entities(real_kg, "kismet") == ["Kismet"]
    marlene_matches = find_matching_entities(real_kg, "[Marlene Dietrich]")
    assert "Marlene Dietrich" in marlene_matches
    dieterle_matches = find_matching_entities(real_kg, "  WILLIAM DIETERLE ")
    assert "William Dieterle" in dieterle_matches

    kismet_out = list(real_kg.graph.edges("Kismet", data=True))
    relations_from_kismet = {data["relation"] for _, _, data in kismet_out}
    assert "directed_by" in relations_from_kismet
    assert "starred_actors" in relations_from_kismet
    assert "written_by" in relations_from_kismet

    directed = [
        tail
        for _, tail, data in kismet_out
        if data["relation"] == "directed_by"
    ]
    assert "William Dieterle" in directed

    starred = [
        tail
        for _, tail, data in kismet_out
        if data["relation"] == "starred_actors"
    ]
    assert "Marlene Dietrich" in starred
    assert "Edward Arnold" in starred


def test_multiple_relations_preserved_on_real_graph(real_kg: KnowledgeGraph) -> None:
    found_pair = None
    for head, tail, data in real_kg.graph.edges(data=True):
        edge_bundle = real_kg.graph.get_edge_data(head, tail) or {}
        rels = {item["relation"] for item in edge_bundle.values()}
        if len(rels) >= 2:
            found_pair = (head, tail, rels)
            break
    assert found_pair is not None, "expected at least one entity pair with 2+ relation types"
    head, tail, rels = found_pair
    assert len(rels) >= 2
    assert real_kg.has_entity(head)
    assert real_kg.has_entity(tail)
