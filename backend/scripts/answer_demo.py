"""Answer a few real MetaQA train questions with the KG engine.

Run from the project root:
    python backend/scripts/answer_demo.py
"""

from __future__ import annotations

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


def _first_qa(path: Path) -> tuple[str, list[str]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return parse_qa_line(line)
    raise ValueError(f"No questions in {path}")


def main() -> None:
    kg = load_graph()
    metaqa = find_metaqa_dir(find_project_root())
    print("ANSWER ENGINE DEMOS (train)")
    print("---------------------------")
    for hops, folder in ((1, "1-hop"), (2, "2-hop"), (3, "3-hop")):
        question, gold = _first_qa(metaqa / folder / "vanilla" / "qa_train.txt")
        result = answer_question(question, max_hops=hops, kg=kg)
        scores = score_answer_sets(result["predicted_answers"], gold)
        print(f"=== {folder} ===")
        print(f"Q: {question}")
        print(f"hop_plan: {result['hop_plan']}")
        print(f"predicted: {result['predicted_answers']}")
        print(f"gold:      {gold}")
        print(f"F1={scores['f1']:.3f} exact={scores['exact_match']} source={result['answer_source']}")
        if result["evidence_paths"]:
            print(f"sample path: {result['evidence_paths'][0]}")
        print()


if __name__ == "__main__":
    main()
