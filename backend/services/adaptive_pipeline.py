"""End-to-end AdaptKG-RAG pipeline: classify, retrieve, rank, answer, explain."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from backend.services.adaptive import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    select_retrieval_depth,
)
from backend.services.answer_engine import answer_question
from backend.services.entity_detection import detect_entities
from backend.services.hop_classifier import load_artifacts, predict_hop
from backend.services.knowledge_graph import KnowledgeGraph, load_graph
from backend.services.ranking import DEFAULT_TOP_K, rank_triples
from backend.services.retrieval import (
    DEFAULT_MAX_CANDIDATES_PER_EXPANSION,
    DEFAULT_MAX_TOTAL_CANDIDATES,
    DEFAULT_PRUNING_STRATEGY,
    retrieve_fixed_hop,
)
from backend.utils.explain import format_evidence_path


def run_adaptive_pipeline(
    question: str,
    kg: KnowledgeGraph | None = None,
    *,
    vectorizer: TfidfVectorizer | None = None,
    model: LogisticRegression | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    hop_prediction: dict[str, Any] | None = None,
    max_candidates_per_expansion: int | None = DEFAULT_MAX_CANDIDATES_PER_EXPANSION,
    max_total_candidates: int | None = DEFAULT_MAX_TOTAL_CANDIDATES,
    pruning_strategy: str = DEFAULT_PRUNING_STRATEGY,
) -> dict[str, Any]:
    """Run the adaptive retrieval + ranking + KG answering pipeline.

    `hop_prediction` may be injected for tests. Otherwise the saved classifier
    is used. Dataset hop labels are never read here.
    """
    graph = kg if kg is not None else load_graph()
    detected = detect_entities(question, kg=graph)
    seeds = [item["name"] for item in detected["entities"] if item.get("in_kg")]

    if hop_prediction is None:
        if vectorizer is None or model is None:
            vectorizer, model = load_artifacts()
        hop_prediction = predict_hop(question, vectorizer=vectorizer, model=model)

    predicted_hop = int(hop_prediction["predicted_hop"])
    confidence = float(hop_prediction["confidence"])
    decision = select_retrieval_depth(
        predicted_hop, confidence, threshold=confidence_threshold
    )
    depth = int(decision["selected_retrieval_depth"])

    retrieval = retrieve_fixed_hop(
        seeds, depth, kg=graph, question=question,
        max_candidates_per_expansion=max_candidates_per_expansion,
        max_total_candidates=max_total_candidates,
        pruning_strategy=pruning_strategy
    )
    ranked = rank_triples(question, retrieval["triples"], top_k=top_k)

    answered = answer_question(question, max_hops=predicted_hop, kg=graph)
    evidence_paths = []
    for item in answered.get("evidence", []):
        path = item.get("path") or []
        evidence_paths.append(
            {
                "answer": item.get("answer"),
                "path": path,
                "explanation": format_evidence_path(path),
            }
        )

    ranked_out = []
    for triple in ranked:
        path = triple.get("path") or []
        ranked_out.append(
            {
                "subject": triple.get("subject"),
                "relation": triple.get("relation"),
                "object": triple.get("object"),
                "depth": triple.get("depth"),
                "direction": triple.get("direction"),
                "relevance_score": triple.get("relevance_score"),
                "evidence_path": path,
                "explanation": format_evidence_path(path),
                "triple_text": triple.get("triple_text"),
            }
        )

    return {
        "question": question,
        "detected_entities": detected["entities"],
        "predicted_hop": predicted_hop,
        "confidence": confidence,
        "probabilities": hop_prediction.get("probabilities", {}),
        "selected_retrieval_depth": depth,
        "adaptive_decision_reason": decision["adaptive_decision_reason"],
        "used_fallback": decision["used_fallback"],
        "candidate_triple_count": retrieval["num_triples"],
        "retrieval_time_ms": retrieval["elapsed_ms"],
        "ranked_triples": ranked_out,
        "answer": answered.get("predicted_answers", []),
        "evidence_paths": evidence_paths,
        "answer_source": answered.get("answer_source", "kg_path"),
        "hop_plan": answered.get("hop_plan", []),
        "seed_entities": seeds,
        "confidence_threshold": confidence_threshold,
    }
