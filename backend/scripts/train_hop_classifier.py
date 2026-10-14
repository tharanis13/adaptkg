"""Train the hop classifier on qa_train.txt and evaluate on qa_test.txt.

Run from the project root:
    python backend/scripts/train_hop_classifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.services.hop_classifier import (
    evaluate_classifier,
    load_questions,
    predict_hop,
    save_artifacts,
    train_classifier,
)
from backend.utils.paths import find_project_root


def _write_report(metrics: dict, path: Path) -> None:
    cm = metrics["confusion_matrix"]
    matrix = cm["matrix"]
    labels = cm["labels"]
    lines = [
        "HOP COMPLEXITY CLASSIFIER - TEST EVALUATION",
        "------------------------------------------",
        "Train split: qa_train.txt from 1-hop, 2-hop, and 3-hop (question text only).",
        "Test split:  qa_test.txt from 1-hop, 2-hop, and 3-hop (combined).",
        "Model: TfidfVectorizer + LogisticRegression.",
        "",
        f"Test examples: {metrics['n_examples']}",
        f"Accuracy:      {metrics['accuracy']:.4f}",
        f"Precision (macro): {metrics['precision_macro']:.4f}",
        f"Recall (macro):    {metrics['recall_macro']:.4f}",
        f"F1-score (macro):  {metrics['f1_macro']:.4f}",
        f"Precision (weighted): {metrics['precision_weighted']:.4f}",
        f"Recall (weighted):    {metrics['recall_weighted']:.4f}",
        f"F1-score (weighted):  {metrics['f1_weighted']:.4f}",
        "",
        "Per-class metrics:",
    ]
    for hop, row in metrics["per_class"].items():
        lines.append(
            f"  hop {hop}: precision={row['precision']:.4f} "
            f"recall={row['recall']:.4f} f1={row['f1']:.4f} support={row['support']}"
        )
    lines.extend(
        [
            "",
            "Confusion matrix (rows=true, cols=pred, order 1/2/3):",
            f"        pred1   pred2   pred3",
        ]
    )
    for label, row in zip(labels, matrix):
        lines.append(f"  true{label}  {row[0]:6d}  {row[1]:6d}  {row[2]:6d}")
    lines.extend(["", "sklearn classification report:", metrics["classification_report"]])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = find_project_root()
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TRAIN questions (qa_train.txt only)...")
    train_texts, train_labels = load_questions("train")
    print(f"  train size: {len(train_texts)}")
    print("  label counts:", {h: train_labels.count(h) for h in (1, 2, 3)})

    print("Training TF-IDF + Logistic Regression...")
    vectorizer, model = train_classifier(train_texts, train_labels)
    saved = save_artifacts(vectorizer, model, root)
    print(f"  saved vectorizer: {saved['vectorizer']}")
    print(f"  saved model:      {saved['model']}")

    print("Loading TEST questions (qa_test.txt only)...")
    test_texts, test_labels = load_questions("test")
    print(f"  test size: {len(test_texts)}")

    print("Evaluating on combined test set...")
    metrics = evaluate_classifier(test_texts, test_labels, vectorizer, model)
    metrics["train_size"] = len(train_texts)
    metrics["artifacts"] = saved

    json_path = results_dir / "hop_classifier_metrics.json"
    txt_path = results_dir / "hop_classifier_metrics.txt"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_report(metrics, txt_path)

    print()
    print(txt_path.read_text(encoding="utf-8"))
    print(f"Metrics JSON: {json_path}")

    demo_q = "which person wrote the films directed by [Yuriy Norshteyn]"
    demo = predict_hop(demo_q, vectorizer=vectorizer, model=model)
    print("Sample prediction:")
    print(json.dumps({"question": demo_q, **demo}, indent=2))


if __name__ == "__main__":
    main()
