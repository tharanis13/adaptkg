"""Rank retrieved KG triples against a question with TF-IDF cosine similarity."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_TOP_K = 10


def triple_to_text(triple: dict[str, Any]) -> str:
    relation = str(triple.get("relation", "")).replace("_", " ")
    return f"{triple.get('subject', '')} {relation} {triple.get('object', '')}".strip()


def rank_triples(
    question: str,
    triples: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    """Return Top-K triples scored by cosine(TF-IDF(question), TF-IDF(triple text))."""
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if not triples:
        return []

    documents = [question] + [triple_to_text(t) for t in triples]
    vectorizer = TfidfVectorizer()
    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        ranked = []
        for triple in triples:
            item = dict(triple)
            item["relevance_score"] = 0.0
            item["triple_text"] = triple_to_text(triple)
            ranked.append(item)
        return ranked[:top_k]

    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    order = np.argsort(-scores)
    ranked: list[dict[str, Any]] = []
    for idx in order[:top_k]:
        item = dict(triples[int(idx)])
        item["relevance_score"] = float(scores[int(idx)])
        item["triple_text"] = triple_to_text(triples[int(idx)])
        ranked.append(item)
    return ranked
