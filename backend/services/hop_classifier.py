"""TF-IDF + Logistic Regression hop-complexity classifier (1/2/3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from backend.scripts.inspect_dataset import parse_qa_line
from backend.utils.paths import find_metaqa_dir, find_project_root

HOP_LABELS = (1, 2, 3)
HOP_FOLDERS = {1: "1-hop", 2: "2-hop", 3: "3-hop"}
VALID_SPLITS = ("train", "dev", "test")


def model_dir(project_root: Path | None = None) -> Path:
    root = project_root or find_project_root()
    path = root / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def vectorizer_path(project_root: Path | None = None) -> Path:
    return model_dir(project_root) / "hop_tfidf_vectorizer.joblib"


def classifier_path(project_root: Path | None = None) -> Path:
    return model_dir(project_root) / "hop_logreg.joblib"


def load_questions(split: str) -> tuple[list[str], list[int]]:
    """Load question text and hop labels. `split` is train, dev, or test."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
    metaqa = find_metaqa_dir()
    texts: list[str] = []
    labels: list[int] = []
    for hop in HOP_LABELS:
        path = metaqa / HOP_FOLDERS[hop] / "vanilla" / f"qa_{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            question, _answers = parse_qa_line(raw)
            texts.append(question)
            labels.append(hop)
    return texts, labels


def train_classifier(
    texts: list[str],
    labels: list[int],
) -> tuple[TfidfVectorizer, LogisticRegression]:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=30000,
        lowercase=True,
    )
    x_train = vectorizer.fit_transform(texts)
    model = LogisticRegression(
        max_iter=300,
        solver="saga",
        class_weight="balanced",
    )
    model.fit(x_train, labels)
    return vectorizer, model


def save_artifacts(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    project_root: Path | None = None,
) -> dict[str, str]:
    v_path = vectorizer_path(project_root)
    m_path = classifier_path(project_root)
    joblib.dump(vectorizer, v_path)
    joblib.dump(model, m_path)
    return {"vectorizer": str(v_path), "model": str(m_path)}


def load_artifacts(
    project_root: Path | None = None,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    v_path = vectorizer_path(project_root)
    m_path = classifier_path(project_root)
    if not v_path.is_file() or not m_path.is_file():
        raise FileNotFoundError(
            f"Missing classifier artifacts. Expected {v_path} and {m_path}."
        )
    return joblib.load(v_path), joblib.load(m_path)


def predict_hop(
    question: str,
    vectorizer: TfidfVectorizer | None = None,
    model: LogisticRegression | None = None,
) -> dict[str, Any]:
    if vectorizer is None or model is None:
        vectorizer, model = load_artifacts()
    x = vectorizer.transform([question])
    proba_row = model.predict_proba(x)[0]
    prob_map = {int(cls): float(p) for cls, p in zip(model.classes_, proba_row)}
    predicted = int(model.classes_[int(np.argmax(proba_row))])
    probabilities = {
        "1": prob_map.get(1, 0.0),
        "2": prob_map.get(2, 0.0),
        "3": prob_map.get(3, 0.0),
    }
    return {
        "predicted_hop": predicted,
        "confidence": float(probabilities[str(predicted)]),
        "probabilities": probabilities,
    }


def evaluate_classifier(
    texts: list[str],
    labels: list[int],
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
) -> dict[str, Any]:
    x = vectorizer.transform(texts)
    y_true = np.array(labels)
    y_pred = model.predict(x)
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(HOP_LABELS), average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(HOP_LABELS), average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=list(HOP_LABELS), average=None, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(HOP_LABELS))
    return {
        "n_examples": int(len(labels)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_m),
        "recall_macro": float(recall_m),
        "f1_macro": float(f1_m),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "per_class": {
            str(hop): {
                "precision": float(per_class[0][i]),
                "recall": float(per_class[1][i]),
                "f1": float(per_class[2][i]),
                "support": int(per_class[3][i]),
            }
            for i, hop in enumerate(HOP_LABELS)
        },
        "confusion_matrix": {
            "labels": [1, 2, 3],
            "matrix": matrix.tolist(),
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=list(HOP_LABELS), digits=4, zero_division=0
        ),
    }
