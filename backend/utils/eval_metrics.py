"""Normalize and score predicted vs gold MetaQA answer sets."""

from __future__ import annotations

from backend.utils.normalize import normalize_entity


def normalize_answer(text: str) -> str:
    return normalize_entity(text)


def score_answer_sets(predicted: list[str], gold: list[str]) -> dict:
    pred_set = {normalize_answer(x) for x in predicted if str(x).strip()}
    gold_set = {normalize_answer(x) for x in gold if str(x).strip()}
    if not pred_set and not gold_set:
        return {
            "exact_match": True,
            "hit": True,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "n_pred": 0,
            "n_gold": 0,
        }
    inter = len(pred_set & gold_set)
    precision = inter / len(pred_set) if pred_set else 0.0
    recall = inter / len(gold_set) if gold_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "exact_match": pred_set == gold_set,
        "hit": inter > 0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": len(pred_set),
        "n_gold": len(gold_set),
    }
