"""Tests for adaptive retrieval, ranking, and evidence explanations."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.scripts.inspect_dataset import parse_qa_line
from backend.services.adaptive import select_retrieval_depth
from backend.services.adaptive_pipeline import run_adaptive_pipeline
from backend.services.hop_classifier import load_artifacts, load_questions
from backend.services.knowledge_graph import KnowledgeGraph, load_graph
from backend.services.ranking import rank_triples, triple_to_text
from backend.utils.explain import format_evidence_path
from backend.utils.paths import find_metaqa_dir, find_project_root


def _first_qa(path: Path) -> tuple[str, list[str]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return parse_qa_line(line)
    raise ValueError(f"No questions in {path}")


def test_select_depth_high_confidence_follows_prediction() -> None:
    for hop in (1, 2, 3):
        decision = select_retrieval_depth(hop, 0.87, threshold=0.60)
        assert decision["selected_retrieval_depth"] == hop
        assert decision["used_fallback"] is False


def test_select_depth_fallback_and_cap_at_3() -> None:
    d1 = select_retrieval_depth(1, 0.40, threshold=0.60)
    assert d1["selected_retrieval_depth"] == 2
    assert d1["used_fallback"] is True
    d2 = select_retrieval_depth(2, 0.10, threshold=0.60)
    assert d2["selected_retrieval_depth"] == 3
    d3 = select_retrieval_depth(3, 0.20, threshold=0.60)
    assert d3["selected_retrieval_depth"] == 3
    assert d3["used_fallback"] is False


def test_ranking_uses_actual_cosine_scores() -> None:
    triples = [
        {
            "subject": "Kismet",
            "relation": "directed_by",
            "object": "William Dieterle",
            "depth": 1,
            "direction": "forward",
            "path": [],
        },
        {
            "subject": "Kismet",
            "relation": "has_genre",
            "object": "Drama",
            "depth": 1,
            "direction": "forward",
            "path": [],
        },
    ]
    ranked = rank_triples("who directed Kismet", triples, top_k=2)
    assert len(ranked) == 2
    assert ranked[0]["relevance_score"] >= ranked[1]["relevance_score"]
    assert "directed" in ranked[0]["triple_text"].lower()
    assert ranked[0]["subject"] == "Kismet"
    assert triple_to_text(triples[0]) == "Kismet directed by William Dieterle"


def test_format_evidence_path_from_real_steps() -> None:
    text = format_evidence_path(
        [
            {
                "subject": "Kismet",
                "relation": "directed_by",
                "object": "William Dieterle",
                "direction": "forward",
                "depth": 1,
            }
        ]
    )
    assert "Kismet" in text
    assert "directed_by" in text
    assert "William Dieterle" in text
    assert "↓" in text


@pytest.fixture(scope="module")
def real_kg() -> KnowledgeGraph:
    return load_graph()


@pytest.fixture(scope="module")
def clf():
    return load_artifacts()


@pytest.fixture(scope="module")
def metaqa_dir() -> Path:
    return find_metaqa_dir(find_project_root())


@pytest.mark.parametrize("folder,true_hop", [("1-hop", 1), ("2-hop", 2), ("3-hop", 3)])
def test_adaptive_pipeline_on_real_train_questions(
    real_kg: KnowledgeGraph,
    clf,
    metaqa_dir: Path,
    folder: str,
    true_hop: int,
) -> None:
    vectorizer, model = clf
    question, _gold = _first_qa(metaqa_dir / folder / "vanilla" / "qa_train.txt")
    result = run_adaptive_pipeline(
        question, kg=real_kg, vectorizer=vectorizer, model=model
    )
    assert result["predicted_hop"] in (1, 2, 3)
    assert result["selected_retrieval_depth"] in (1, 2, 3)
    assert result["selected_retrieval_depth"] <= 3
    if result["confidence"] >= 0.60:
        assert result["selected_retrieval_depth"] == result["predicted_hop"]
        assert result["used_fallback"] is False
    else:
        expected = min(result["predicted_hop"] + 1, 3)
        assert result["selected_retrieval_depth"] == expected
    assert result["candidate_triple_count"] >= 0
    assert all(t["depth"] <= result["selected_retrieval_depth"] for t in result["ranked_triples"])
    assert all("relevance_score" in t for t in result["ranked_triples"])
    assert result["answer_source"] == "kg_path"
    for item in result["evidence_paths"]:
        assert "explanation" in item
        if item["path"]:
            assert item["explanation"]
            assert "↓" in item["explanation"]


def test_injected_low_confidence_triggers_fallback(real_kg: KnowledgeGraph) -> None:
    question = "who directed [Kismet]"
    result = run_adaptive_pipeline(
        question,
        kg=real_kg,
        hop_prediction={
            "predicted_hop": 1,
            "confidence": 0.41,
            "probabilities": {"1": 0.41, "2": 0.30, "3": 0.29},
        },
        confidence_threshold=0.60,
    )
    assert result["predicted_hop"] == 1
    assert result["selected_retrieval_depth"] == 2
    assert result["used_fallback"] is True
    assert result["candidate_triple_count"] > 0
    assert any(t["depth"] <= 2 for t in result["ranked_triples"])
    assert all(t["depth"] <= 2 for t in result["ranked_triples"])
    assert result["answer"]  # answer engine still runs
    assert any("Kismet" in (item.get("explanation") or "") for item in result["evidence_paths"])


def test_scan_real_test_questions_for_low_conf_and_mistakes(clf) -> None:
    vectorizer, model = clf
    texts, labels = load_questions("test")
    low = []
    mistakes = []
    matrix = vectorizer.transform(texts)
    preds = model.predict(matrix)
    proba = model.predict_proba(matrix)
    for question, true_hop, pred_hop, row in zip(texts, labels, preds, proba):
        confidence = float(row.max())
        predicted = int(pred_hop)
        if confidence < 0.60:
            low.append((true_hop, question, predicted, confidence))
        if predicted != true_hop:
            mistakes.append((true_hop, question, predicted, confidence))
    for true_hop, question, predicted, confidence in mistakes[:5]:
        print(
            f"MISTAKE true={true_hop} pred={predicted} "
            f"conf={confidence:.4f} Q={question}"
        )
    for true_hop, question, predicted, confidence in low[:5]:
        print(
            f"LOW CONF true={true_hop} pred={predicted} "
            f"conf={confidence:.4f} Q={question}"
        )
    print(f"scanned={len(texts)} low_confidence={len(low)} mistakes={len(mistakes)}")
