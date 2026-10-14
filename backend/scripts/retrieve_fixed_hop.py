"""Demo fixed-hop retrieval on real MetaQA train questions.

Run from the project root:
    python backend/scripts/retrieve_fixed_hop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.scripts.inspect_dataset import parse_qa_line
from backend.services.knowledge_graph import load_graph
from backend.services.retrieval import retrieve_for_question
from backend.utils.paths import find_metaqa_dir, find_project_root


def _first_question(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            question, _answers = parse_qa_line(line)
            return question
    raise ValueError(f"No questions in {path}")


def _print_result(result: dict, sample_n: int = 3) -> None:
    print(f"Question: {result['question']}")
    print(f"Detected entities: {result['detected_entities']}")
    print(f"Requested max hops: {result['max_hops']}")
    print(f"Seed entities: {result['seed_entities']}")
    print(f"Retrieved triple count: {result['num_triples']}")
    print(f"Retrieval time (ms): {result['elapsed_ms']:.2f}")
    depths = {t["depth"] for t in result["triples"]}
    print(f"Observed depths: {sorted(depths) if depths else []}")
    print("Sample retrieved triples:")
    for triple in result["triples"][:sample_n]:
        print(
            f"  ({triple['subject']}) -[{triple['relation']}]-> "
            f"({triple['object']})  depth={triple['depth']} "
            f"direction={triple['direction']}"
        )
        path_txt = " ; ".join(
            f"{step['subject']}|{step['relation']}|{step['object']}"
            f"[{step['direction']}/d{step['depth']}]"
            for step in triple["path"]
        )
        print(f"    path: {path_txt}")
    print()


def main() -> None:
    kg = load_graph()
    metaqa = find_metaqa_dir(find_project_root())
    cases = [
        (1, metaqa / "1-hop" / "vanilla" / "qa_train.txt"),
        (2, metaqa / "2-hop" / "vanilla" / "qa_train.txt"),
        (3, metaqa / "3-hop" / "vanilla" / "qa_train.txt"),
    ]
    print("FIXED-HOP RETRIEVAL DEMOS")
    print("-------------------------")
    print()
    for max_hops, path in cases:
        print(f"=== {max_hops}-hop train sample ===")
        question = _first_question(path)
        result = retrieve_for_question(question, max_hops, kg=kg)
        _print_result(result)


if __name__ == "__main__":
    main()
