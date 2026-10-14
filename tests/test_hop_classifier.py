"""Tests for the hop-complexity classifier (no test-set training)."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.services.hop_classifier import (
    evaluate_classifier,
    load_questions,
    predict_hop,
    train_classifier,
)


def test_tiny_tfidf_logreg_predict_shape() -> None:
    texts = [
        "who directed ENT",
        "who starred in ENT",
        "when was ENT released",
        "which movies have the same director of ENT",
        "who directed the movies acted by ENT",
        "the actor of ENT also starred in which movies",
        "films that share actors with ENT were released in which years",
        "who directed movies that share actors with ENT",
        "what types are the films directed by the director of ENT",
    ]
    labels = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    vectorizer, model = train_classifier(texts, labels)
    result = predict_hop("who directed [Kismet]", vectorizer=vectorizer, model=model)
    assert result["predicted_hop"] in (1, 2, 3)
    assert 0.0 <= result["confidence"] <= 1.0
    probs = result["probabilities"]
    assert set(probs) == {"1", "2", "3"}
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert abs(result["confidence"] - probs[str(result["predicted_hop"])]) < 1e-9


def test_evaluate_returns_confusion_matrix() -> None:
    texts = ["who directed ENT"] * 4 + ["same director of ENT"] * 4 + ["share actors with ENT years"] * 4
    labels = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    vectorizer, model = train_classifier(texts, labels)
    metrics = evaluate_classifier(texts, labels, vectorizer, model)
    assert metrics["n_examples"] == 12
    assert "accuracy" in metrics
    assert metrics["confusion_matrix"]["labels"] == [1, 2, 3]
    assert len(metrics["confusion_matrix"]["matrix"]) == 3


def test_load_train_does_not_read_test_paths(monkeypatch, tmp_path: Path) -> None:
    """Guard: load_questions('train') only requests qa_train.txt files."""
    requested: list[str] = []
    original = Path.read_text

    def fake_read_text(self, *args, **kwargs):
        requested.append(self.name)
        if self.name.startswith("qa_"):
            return "who directed [X]\tY\n"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    # Skip if real dataset missing; this test patches reads of whatever load_questions opens.
    texts, labels = load_questions("train")
    assert texts
    assert set(labels) == {1, 2, 3}
    assert all(name == "qa_train.txt" for name in requested)
    assert "qa_test.txt" not in requested
    assert "qa_dev.txt" not in requested
