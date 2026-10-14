"""Tests for Phase 4 fixed-hop retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.scripts.inspect_dataset import parse_qa_line
from backend.services.knowledge_graph import KnowledgeGraph, load_graph
from backend.services.retrieval import (
    retrieve_fixed_hop,
    retrieve_for_question,
)
from backend.utils.paths import find_metaqa_dir, find_project_root


def _tiny_kg() -> KnowledgeGraph:
    return KnowledgeGraph.from_triples(
        [
            ("MovieA", "starred_actors", "ActorX"),
            ("MovieA", "directed_by", "DirY"),
            ("MovieB", "starred_actors", "ActorX"),
            ("MovieB", "has_genre", "Drama"),
            ("MovieC", "directed_by", "DirY"),
            ("MovieC", "in_language", "English"),
        ]
    )


def _first_question(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            question, _answers = parse_qa_line(line)
            return question
    raise ValueError(f"No questions in {path}")


def test_forward_and_reverse_preserve_original_triple() -> None:
    kg = _tiny_kg()
    forward = retrieve_fixed_hop(["MovieA"], 1, kg=kg)
    assert any(
        t["subject"] == "MovieA"
        and t["relation"] == "starred_actors"
        and t["object"] == "ActorX"
        and t["direction"] == "forward"
        and t["depth"] == 1
        for t in forward["triples"]
    )

    reverse = retrieve_fixed_hop(["ActorX"], 1, kg=kg)
    match = next(
        t
        for t in reverse["triples"]
        if t["subject"] == "MovieA" and t["relation"] == "starred_actors"
    )
    assert match["object"] == "ActorX"
    assert match["direction"] == "reverse"
    assert match["depth"] == 1
    assert match["path"][0]["direction"] == "reverse"


def test_depth_never_exceeds_max_hops() -> None:
    kg = _tiny_kg()
    for hops in (1, 2, 3):
        result = retrieve_fixed_hop(["ActorX"], hops, kg=kg)
        assert result["max_hops"] == hops
        assert all(t["depth"] <= hops for t in result["triples"])
        assert all(step["depth"] <= hops for t in result["triples"] for step in t["path"])
        assert result["elapsed_ms"] >= 0


def test_duplicate_triples_kept_once() -> None:
    kg = KnowledgeGraph.from_triples(
        [
            ("MovieA", "starred_actors", "ActorX"),
            ("MovieA", "starred_actors", "ActorX"),
        ]
    )
    result = retrieve_fixed_hop(["MovieA"], 1, kg=kg)
    keys = [(t["subject"], t["relation"], t["object"]) for t in result["triples"]]
    assert keys.count(("MovieA", "starred_actors", "ActorX")) == 1


def test_cycle_does_not_loop_forever() -> None:
    kg = KnowledgeGraph.from_triples(
        [
            ("A", "r", "B"),
            ("B", "r", "A"),
        ]
    )
    result = retrieve_fixed_hop(["A"], 3, kg=kg)
    assert result["num_triples"] == 2
    assert all(t["depth"] <= 3 for t in result["triples"])


def test_invalid_hops_rejected() -> None:
    kg = _tiny_kg()
    with pytest.raises(ValueError):
        retrieve_fixed_hop(["MovieA"], 4, kg=kg)


@pytest.fixture(scope="module")
def real_kg() -> KnowledgeGraph:
    return load_graph()


@pytest.fixture(scope="module")
def metaqa_dir() -> Path:
    return find_metaqa_dir(find_project_root())


def test_real_kismet_forward_and_dietrich_reverse(real_kg: KnowledgeGraph) -> None:
    one = retrieve_fixed_hop(["Kismet"], 1, kg=real_kg)
    assert one["num_triples"] > 0
    assert any(
        t["subject"] == "Kismet"
        and t["relation"] == "directed_by"
        and t["object"] == "William Dieterle"
        and t["direction"] == "forward"
        for t in one["triples"]
    )

    reverse = retrieve_fixed_hop(["Marlene Dietrich"], 1, kg=real_kg)
    assert any(
        t["subject"] == "Kismet"
        and t["relation"] == "starred_actors"
        and t["object"] == "Marlene Dietrich"
        and t["direction"] == "reverse"
        for t in reverse["triples"]
    )


@pytest.mark.parametrize(
    "hop_folder,max_hops,snippet",
    [
        ("1-hop", 1, "[ginger rogers]"),
        ("2-hop", 2, "[Yuriy Norshteyn]"),
        ("3-hop", 3, "[Dil Chahta Hai]"),
    ],
)
def test_real_train_questions_all_hop_categories(
    real_kg: KnowledgeGraph,
    metaqa_dir: Path,
    hop_folder: str,
    max_hops: int,
    snippet: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    question = _first_question(metaqa_dir / hop_folder / "vanilla" / "qa_train.txt")
    assert snippet in question
    result = retrieve_for_question(question, max_hops, kg=real_kg)
    assert result["num_triples"] > 0
    assert result["seed_entities"]
    assert all(t["depth"] <= max_hops for t in result["triples"])
    assert all(len(t["path"]) == t["depth"] for t in result["triples"])
    assert result["elapsed_ms"] >= 0

    print(f"Question: {result['question']}")
    print(f"Detected entities: {result['detected_entities']}")
    print(f"Requested max hops: {result['max_hops']}")
    print(f"Retrieved triple count: {result['num_triples']}")
    print(f"Retrieval time (ms): {result['elapsed_ms']:.2f}")
    for triple in result["triples"][:2]:
        print(
            f"Sample: {triple['subject']}|{triple['relation']}|{triple['object']} "
            f"depth={triple['depth']} dir={triple['direction']}"
        )
        print(f"Path: {triple['path']}")
    captured = capsys.readouterr()
    assert "Retrieved triple count" in captured.out


def test_pruning_limits_candidates() -> None:
    kg = KnowledgeGraph.from_triples(
        [
            ("A", "directed_by", "B1"),
            ("A", "directed_by", "B2"),
            ("A", "directed_by", "B3"),
            ("A", "directed_by", "B4"),
            ("A", "directed_by", "B5"),
        ]
    )
    # With pruning strategy relevance_bounded and max_candidates_per_expansion=2
    res = retrieve_fixed_hop(
        ["A"], 1, kg=kg,
        max_candidates_per_expansion=2,
        pruning_strategy="relevance_bounded"
    )
    # Expanding A should only produce 2 triples because of bounding
    assert res["num_triples"] == 2
    # Check that paths are preserved
    for t in res["triples"]:
        assert len(t["path"]) == 1
        assert t["path"][0]["subject"] == "A"


def test_pruning_determinism() -> None:
    kg = KnowledgeGraph.from_triples(
        [
            ("A", "directed_by", "B4"),
            ("A", "directed_by", "B1"),
            ("A", "directed_by", "B3"),
            ("A", "directed_by", "B2"),
        ]
    )
    # Relevance bounded sorting should break ties alphabetically
    res1 = retrieve_fixed_hop(
        ["A"], 1, kg=kg,
        question="who is the director of A?",
        max_candidates_per_expansion=2,
        pruning_strategy="relevance_bounded"
    )
    res2 = retrieve_fixed_hop(
        ["A"], 1, kg=kg,
        question="who is the director of A?",
        max_candidates_per_expansion=2,
        pruning_strategy="relevance_bounded"
    )
    # Results must be identical
    triples1 = sorted([(t["subject"], t["relation"], t["object"]) for t in res1["triples"]])
    triples2 = sorted([(t["subject"], t["relation"], t["object"]) for t in res2["triples"]])
    assert triples1 == triples2
    assert len(triples1) == 2


def test_hub_pruning(real_kg: KnowledgeGraph) -> None:
    # "English" is a huge hub node. Doing a 2-hop search from English unpruned would explode.
    # We run it with relevance_bounded pruning and max_candidates_per_expansion=5, max_total_candidates=50.
    res = retrieve_fixed_hop(
        ["English"], 2, kg=real_kg,
        question="what movies are in English?",
        max_candidates_per_expansion=5,
        max_total_candidates=50,
        pruning_strategy="relevance_bounded"
    )
    assert res["num_triples"] <= 50
    assert all(t["depth"] <= 2 for t in res["triples"])
    assert all(len(t["path"]) == t["depth"] for t in res["triples"])

