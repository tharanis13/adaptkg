"""
Phase 1: Inspect the local MetaQA dataset without modifying it.

Run from the project root:
    python backend/scripts/inspect_dataset.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.utils.paths import find_metaqa_dir, find_project_root

HOP_NAMES = ("1-hop", "2-hop", "3-hop")
SPLITS = ("train", "dev", "test")
ENTITY_PATTERN = re.compile(r"\[([^\]]+)\]")


def read_nonempty_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [line.rstrip("\n\r") for line in text.splitlines() if line.strip()]


def parse_qa_line(line: str) -> tuple[str, list[str]]:
    if "\t" not in line:
        raise ValueError(f"QA line is missing a tab separator: {line[:120]!r}")
    question, answers_field = line.split("\t", 1)
    answers = [a for a in answers_field.split("|") if a != ""]
    return question, answers


def inspect_kb(kb_path: Path) -> dict:
    triples: list[tuple[str, str, str]] = []
    malformed = 0
    for line in read_nonempty_lines(kb_path):
        parts = line.split("|")
        if len(parts) != 3:
            malformed += 1
            continue
        head, relation, tail = parts
        triples.append((head, relation, tail))

    entities = {h for h, _, t in triples} | {t for _, _, t in triples}
    relations = {r for _, r, _ in triples}
    relation_counts = Counter(r for _, r, _ in triples)
    return {
        "path": kb_path,
        "triples": triples,
        "num_triples": len(triples),
        "malformed": malformed,
        "num_entities": len(entities),
        "num_relations": len(relations),
        "relations_sorted": sorted(relations),
        "relation_counts": relation_counts,
        "samples": triples[:5],
    }


def inspect_qa_file(path: Path) -> dict:
    lines = read_nonempty_lines(path)
    questions: list[str] = []
    answer_lists: list[list[str]] = []
    entity_span_counts: list[int] = []
    multi_answer = 0
    missing_tab = 0

    for line in lines:
        try:
            question, answers = parse_qa_line(line)
        except ValueError:
            missing_tab += 1
            continue
        questions.append(question)
        answer_lists.append(answers)
        entity_span_counts.append(len(ENTITY_PATTERN.findall(question)))
        if len(answers) > 1:
            multi_answer += 1

    samples = []
    for q, a in zip(questions[:3], answer_lists[:3]):
        samples.append((q, a, ENTITY_PATTERN.findall(q)))

    return {
        "path": path,
        "count": len(questions),
        "multi_answer": multi_answer,
        "missing_tab": missing_tab,
        "entity_span_min": min(entity_span_counts) if entity_span_counts else 0,
        "entity_span_max": max(entity_span_counts) if entity_span_counts else 0,
        "samples": samples,
    }


def qa_path(metaqa_dir: Path, hop: str, split: str) -> Path:
    return metaqa_dir / hop / "vanilla" / f"qa_{split}.txt"


def print_report(project_root: Path, metaqa_dir: Path, kb: dict, qa: dict) -> None:
    print("DATASET INSPECTION REPORT")
    print("-------------------------")
    print(f"Project root: {project_root}")
    print(f"Dataset dir:  {metaqa_dir}")
    print(f"KG file:      {kb['path']}")
    print()
    print(f"KG triples: {kb['num_triples']}")
    print(f"Unique entities: {kb['num_entities']}")
    print(f"Unique relation types: {kb['num_relations']}")
    if kb["malformed"]:
        print(f"Malformed KG lines (skipped): {kb['malformed']}")
    print()
    print("Relation types:")
    for rel in kb["relations_sorted"]:
        print(f"  - {rel} ({kb['relation_counts'][rel]})")
    print()

    total_questions = 0
    for hop in HOP_NAMES:
        print(hop.upper())
        for split in SPLITS:
            count = qa[hop][split]["count"]
            total_questions += count
            label = split.capitalize()
            print(f"{label}: {count}")
        print()

    print(f"TOTAL QUESTIONS: {total_questions}")
    print()

    print("FORMAT NOTES")
    print("------------")
    print("KG triple: head|relation|tail")
    print("QA line:   question<TAB>answer1|answer2|...")
    print("Entity:    marked in the question with square brackets, e.g. [ginger rogers]")
    print()

    print("SAMPLE KG TRIPLES")
    print("-----------------")
    for head, rel, tail in kb["samples"]:
        print(f"{head}|{rel}|{tail}")
    print()

    print("SAMPLE QA EXAMPLES")
    print("------------------")
    for hop in HOP_NAMES:
        for split in SPLITS:
            info = qa[hop][split]
            print(f"[{hop} / {split}] file: {info['path'].relative_to(project_root)}")
            print(
                f"  multi-answer questions: {info['multi_answer']} / {info['count']}; "
                f"bracketed entities per question: min={info['entity_span_min']} "
                f"max={info['entity_span_max']}"
            )
            if info["missing_tab"]:
                print(f"  lines missing tab separator: {info['missing_tab']}")
            for question, answers, entities in info["samples"]:
                print(f"  Q: {question}")
                print(f"     entities: {entities}")
                print(f"     answers ({len(answers)}): {' | '.join(answers[:8])}"
                      + (" ..." if len(answers) > 8 else ""))
            print()


def main() -> None:
    project_root = find_project_root()
    metaqa_dir = find_metaqa_dir(project_root)
    kb = inspect_kb(metaqa_dir / "kb.txt")

    qa: dict[str, dict[str, dict]] = {}
    missing_files: list[Path] = []
    for hop in HOP_NAMES:
        qa[hop] = {}
        for split in SPLITS:
            path = qa_path(metaqa_dir, hop, split)
            if not path.is_file():
                missing_files.append(path)
                qa[hop][split] = {
                    "path": path,
                    "count": 0,
                    "multi_answer": 0,
                    "missing_tab": 0,
                    "entity_span_min": 0,
                    "entity_span_max": 0,
                    "samples": [],
                }
            else:
                qa[hop][split] = inspect_qa_file(path)

    print_report(project_root, metaqa_dir, kb, qa)
    if missing_files:
        print("WARNING: missing QA files:")
        for path in missing_files:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
