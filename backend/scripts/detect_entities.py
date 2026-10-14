"""Run entity detection on a few real MetaQA questions.

Run from the project root:
    python backend/scripts/detect_entities.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.scripts.inspect_dataset import parse_qa_line
from backend.services.entity_detection import detect_entities
from backend.services.knowledge_graph import load_graph
from backend.utils.paths import find_metaqa_dir, find_project_root


def _first_question(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            question, _answers = parse_qa_line(line)
            return question
    raise ValueError(f"No questions in {path}")


def main() -> None:
    kg = load_graph()
    metaqa = find_metaqa_dir(find_project_root())
    samples = [
        _first_question(metaqa / "1-hop" / "vanilla" / "qa_train.txt"),
        _first_question(metaqa / "2-hop" / "vanilla" / "qa_train.txt"),
        _first_question(metaqa / "3-hop" / "vanilla" / "qa_train.txt"),
        "what movies are about ginger rogers",
        "who directed KISMET",
        "compare [Kismet] and [The Sentinel]",
        "who directed [Totally Unknown Film XYZ]",
    ]
    print("ENTITY DETECTION DEMOS")
    print("----------------------")
    for question in samples:
        result = detect_entities(question, kg=kg)
        print(f"Q: {question}")
        print(json.dumps(result, ensure_ascii=True, indent=2))
        print()


if __name__ == "__main__":
    main()
