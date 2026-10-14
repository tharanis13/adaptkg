"""Tests for the Phase 5 deterministic answer engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.scripts.inspect_dataset import parse_qa_line
from backend.services.answer_engine import answer_question, parse_hop_plan
from backend.services.knowledge_graph import KnowledgeGraph, load_graph
from backend.utils.eval_metrics import score_answer_sets
from backend.utils.paths import find_metaqa_dir, find_project_root


def _tiny_kg() -> KnowledgeGraph:
    return KnowledgeGraph.from_triples(
        [
            ("MovieA", "directed_by", "DirY"),
            ("MovieB", "directed_by", "DirY"),
            ("MovieA", "starred_actors", "ActorX"),
            ("MovieB", "starred_actors", "ActorX"),
            ("MovieB", "has_genre", "Drama"),
        ]
    )


def _first_qa(path: Path) -> tuple[str, list[str]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return parse_qa_line(line)
    raise ValueError(f"No questions in {path}")


def test_same_director_excludes_seed() -> None:
    kg = _tiny_kg()
    result = answer_question(
        "which movies have the same director of [MovieA]",
        max_hops=2,
        kg=kg,
    )
    assert result["answer_source"] == "kg_path"
    assert "MovieB" in result["predicted_answers"]
    assert "MovieA" not in result["predicted_answers"]
    assert result["evidence_paths"]


def test_forward_director_on_synthetic_graph() -> None:
    kg = _tiny_kg()
    result = answer_question("who directed [MovieA]", max_hops=1, kg=kg)
    assert result["predicted_answers"] == ["DirY"]


def test_unknown_seed_returns_empty() -> None:
    kg = _tiny_kg()
    result = answer_question("who directed [Missing Film]", max_hops=1, kg=kg)
    assert result["predicted_answers"] == []
    assert result["answer_source"] == "kg_path"


def test_parse_plans_match_expected_relations() -> None:
    p1 = parse_hop_plan("what movies are about [ginger rogers]", 1)
    assert p1[0]["relation"] == "has_tags"
    assert p1[0]["direction"] == "reverse"
    p2 = parse_hop_plan("which person wrote the films directed by [Yuriy Norshteyn]", 2)
    assert [h["relation"] for h in p2] == ["directed_by", "written_by"]
    p3 = parse_hop_plan(
        "the films that share actors with the film [Dil Chahta Hai] were released in which years",
        3,
    )
    assert [h["relation"] for h in p3] == [
        "starred_actors",
        "starred_actors",
        "release_year",
    ]


@pytest.fixture(scope="module")
def real_kg() -> KnowledgeGraph:
    return load_graph()


@pytest.fixture(scope="module")
def metaqa_dir() -> Path:
    return find_metaqa_dir(find_project_root())


@pytest.mark.parametrize(
    "folder,hops",
    [("1-hop", 1), ("2-hop", 2), ("3-hop", 3)],
)
def test_real_train_first_question(
    real_kg: KnowledgeGraph,
    metaqa_dir: Path,
    folder: str,
    hops: int,
) -> None:
    question, gold = _first_qa(metaqa_dir / folder / "vanilla" / "qa_train.txt")
    result = answer_question(question, max_hops=hops, kg=real_kg)
    scores = score_answer_sets(result["predicted_answers"], gold)
    assert result["predicted_answers"], f"no answers for: {question}"
    assert scores["hit"], (
        f"missed gold for {question!r}: pred={result['predicted_answers'][:8]!r} gold={gold[:8]!r}"
    )
    assert result["answer_source"] == "kg_path"
    assert result["evidence_paths"]
