"""Evaluate the KG answer engine on a sample of each MetaQA test split.

Run from the project root:
    python backend/scripts/evaluate_answers.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.scripts.inspect_dataset import parse_qa_line
from backend.services.answer_engine import answer_question
from backend.services.knowledge_graph import load_graph
from backend.utils.eval_metrics import score_answer_sets
from backend.utils.paths import find_metaqa_dir, find_project_root

SAMPLE_PER_SPLIT = 50
RNG_SEED = 42
HOPS = (("1-hop", 1), ("2-hop", 2), ("3-hop", 3))


def _load_qa(path: Path) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(parse_qa_line(line))
    return rows


def _sample(rows: list[tuple[str, list[str]]], k: int, seed: int) -> list[tuple[str, list[str]]]:
    rng = random.Random(seed)
    if len(rows) <= k:
        return list(rows)
    return rng.sample(rows, k)


def main() -> None:
    kg = load_graph()
    metaqa = find_metaqa_dir(find_project_root())
    out_dir = find_project_root() / "results"
    out_dir.mkdir(exist_ok=True)
    failure_path = out_dir / "phase5_failures.jsonl"
    report_path = out_dir / "phase5_eval_report.txt"

    all_rows: list[str] = []
    failures: list[dict] = []
    successes: list[dict] = []
    totals = {"tested": 0, "correct": 0, "incorrect": 0, "hits": 0, "f1_sum": 0.0}

    print("PHASE 5 TEST-SPLIT SAMPLE EVALUATION")
    print("------------------------------------")
    print(f"sample_per_split={SAMPLE_PER_SPLIT}  seed={RNG_SEED}")
    print()

    with failure_path.open("w", encoding="utf-8") as fail_f:
        for folder, hops in HOPS:
            path = metaqa / folder / "vanilla" / "qa_test.txt"
            sample = _sample(_load_qa(path), SAMPLE_PER_SPLIT, RNG_SEED + hops)
            correct = 0
            hit_n = 0
            f1_sum = 0.0
            split_success: list[dict] = []
            split_fail: list[dict] = []
            for question, gold in sample:
                result = answer_question(question, max_hops=hops, kg=kg)
                scores = score_answer_sets(result["predicted_answers"], gold)
                totals["tested"] += 1
                totals["f1_sum"] += scores["f1"]
                rec = {
                    "split": folder,
                    "max_hops": hops,
                    "question": question,
                    "expected": gold,
                    "predicted": result["predicted_answers"],
                    "hop_plan": result["hop_plan"],
                    "evidence": result["evidence"][:5],
                    "exact_match": scores["exact_match"],
                    "hit": scores["hit"],
                    "f1": scores["f1"],
                }
                if scores["hit"]:
                    hit_n += 1
                    totals["hits"] += 1
                if scores["exact_match"]:
                    correct += 1
                    totals["correct"] += 1
                    split_success.append(rec)
                    successes.append(rec)
                else:
                    totals["incorrect"] += 1
                    split_fail.append(rec)
                    failures.append(rec)
                    fail_f.write(json.dumps(rec, ensure_ascii=True) + "\n")

            n = len(sample)
            acc = correct / n if n else 0.0
            mean_f1 = f1_sum / n if n else 0.0
            hit_acc = hit_n / n if n else 0.0
            line = (
                f"{folder}: tested={n} correct={correct} incorrect={n - correct} "
                f"accuracy={acc:.3f} hits={hit_acc:.3f} mean_F1={mean_f1:.3f}"
            )
            print(line)
            all_rows.append(line)
            if split_success:
                s = split_success[0]
                print(f"  success e.g. Q: {s['question']}")
                print(f"    expected: {s['expected'][:8]}")
                print(f"    predicted: {s['predicted'][:8]}")
            if split_fail:
                f = split_fail[0]
                print(f"  failure e.g. Q: {f['question']}")
                print(f"    expected: {f['expected'][:8]}")
                print(f"    predicted: {f['predicted'][:8]}")
                print(f"    hop_plan: {f['hop_plan']}")
            print()

    n = totals["tested"]
    acc = totals["correct"] / n if n else 0.0
    mean_f1 = totals["f1_sum"] / n if n else 0.0
    hit_acc = totals["hits"] / n if n else 0.0
    summary = (
        f"TOTAL: tested={n} correct={totals['correct']} incorrect={totals['incorrect']} "
        f"accuracy={acc:.3f} hits={hit_acc:.3f} mean_F1={mean_f1:.3f}"
    )
    print(summary)
    print(f"Failure log: {failure_path}")

    report = "\n".join(
        [
            "PHASE 5 EVALUATION REPORT",
            "-------------------------",
            f"Exact-match accuracy is the primary 'correct' count.",
            f"Hits = at least one gold answer predicted. mean_F1 is set overlap.",
            f"sample_per_split={SAMPLE_PER_SPLIT} seed={RNG_SEED}",
            "",
            *all_rows,
            summary,
            "",
            "Successful examples:",
        ]
    )
    for rec in successes[:3]:
        report += (
            f"\n- [{rec['split']}] {rec['question']}\n"
            f"  expected: {rec['expected']}\n"
            f"  predicted: {rec['predicted']}\n"
        )
    report += "\nFailure examples:\n"
    for rec in failures[:5]:
        report += (
            f"\n- [{rec['split']}] {rec['question']}\n"
            f"  expected: {rec['expected']}\n"
            f"  predicted: {rec['predicted']}\n"
            f"  hop_plan: {rec['hop_plan']}\n"
            f"  evidence (up to 5): {json.dumps(rec['evidence'], ensure_ascii=True)}\n"
        )
    report_path.write_text(report, encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
