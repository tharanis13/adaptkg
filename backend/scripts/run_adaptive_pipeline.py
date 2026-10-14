"""Run the adaptive AdaptKG-RAG pipeline on real MetaQA questions.

Run from the project root:
    python backend/scripts/run_adaptive_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.scripts.inspect_dataset import parse_qa_line
from backend.services.adaptive_pipeline import run_adaptive_pipeline
from backend.services.hop_classifier import load_artifacts, load_questions
from backend.services.knowledge_graph import load_graph
from backend.utils.paths import find_metaqa_dir, find_project_root


def _first_question(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            question, _answers = parse_qa_line(line)
            return question
    raise ValueError(f"No questions in {path}")


def _summarize(result: dict) -> dict:
    return {
        "question": result["question"],
        "detected_entities": result["detected_entities"],
        "predicted_hop": result["predicted_hop"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "selected_retrieval_depth": result["selected_retrieval_depth"],
        "adaptive_decision_reason": result["adaptive_decision_reason"],
        "candidate_triple_count": result["candidate_triple_count"],
        "retrieval_time_ms": round(result["retrieval_time_ms"], 2),
        "ranked_triples": [
            {
                "subject": t["subject"],
                "relation": t["relation"],
                "object": t["object"],
                "depth": t["depth"],
                "direction": t["direction"],
                "relevance_score": t["relevance_score"],
                "explanation": t["explanation"],
            }
            for t in result["ranked_triples"][:3]
        ],
        "answer": result["answer"][:12],
        "evidence_paths": result["evidence_paths"][:2],
    }


def main() -> None:
    kg = load_graph()
    vectorizer, model = load_artifacts()
    metaqa = find_metaqa_dir(find_project_root())

    print("ADAPTIVE PIPELINE DEMOS")
    print("-----------------------")
    for hops, folder in ((1, "1-hop"), (2, "2-hop"), (3, "3-hop")):
        question = _first_question(metaqa / folder / "vanilla" / "qa_train.txt")
        result = run_adaptive_pipeline(
            question, kg=kg, vectorizer=vectorizer, model=model
        )
        print(f"=== {folder} train sample ===")
        print(json.dumps(_summarize(result), ensure_ascii=True, indent=2))
        print()

    print("SCAN FULL TEST SET FOR LOW CONFIDENCE OR MISTAKES")
    print("------------------------------------------------")
    texts, labels = load_questions("test")
    matrix = vectorizer.transform(texts)
    preds = model.predict(matrix)
    proba = model.predict_proba(matrix)
    found_low = []
    found_mistakes = []
    for question, true_hop, pred_hop, row in zip(texts, labels, preds, proba):
        confidence = float(row.max())
        predicted_hop = int(pred_hop)
        record = {
            "true_hop": int(true_hop),
            "question": question,
            "predicted_hop": predicted_hop,
            "confidence": confidence,
        }
        if confidence < 0.60 and len(found_low) < 5:
            found_low.append(record)
        if predicted_hop != int(true_hop) and len(found_mistakes) < 5:
            found_mistakes.append(record)

    print("low_confidence_examples:")
    print(json.dumps(found_low, ensure_ascii=True, indent=2))
    print("classifier_mistake_examples:")
    print(json.dumps(found_mistakes, ensure_ascii=True, indent=2))

    if found_mistakes:
        sample = found_mistakes[0]
        result = run_adaptive_pipeline(
            sample["question"], kg=kg, vectorizer=vectorizer, model=model
        )
        print("pipeline on a classifier mistake:")
        print(json.dumps(_summarize(result), ensure_ascii=True, indent=2))
    if found_low:
        sample = found_low[0]
        result = run_adaptive_pipeline(
            sample["question"], kg=kg, vectorizer=vectorizer, model=model
        )
        print("pipeline on a low-confidence question:")
        print(json.dumps(_summarize(result), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
