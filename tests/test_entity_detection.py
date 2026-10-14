"""Tests for Phase 3 entity detection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.scripts.inspect_dataset import parse_qa_line
from backend.services.entity_detection import detect_entities
from backend.services.knowledge_graph import KnowledgeGraph, load_graph
from backend.utils.paths import find_metaqa_dir, find_project_root


def _first_question(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            question, _answers = parse_qa_line(line)
            return question
    raise ValueError(f"No questions in {path}")


@pytest.fixture(scope="module")
def kg() -> KnowledgeGraph:
    return load_graph()


@pytest.fixture(scope="module")
def metaqa_dir() -> Path:
    return find_metaqa_dir(find_project_root())


def test_real_1hop_bracket_question(kg: KnowledgeGraph, metaqa_dir: Path) -> None:
    question = _first_question(metaqa_dir / "1-hop" / "vanilla" / "qa_train.txt")
    assert "[ginger rogers]" in question
    result = detect_entities(question, kg=kg)
    names = [item["name"] for item in result["entities"]]
    methods = {item["method"] for item in result["entities"]}
    assert any("ginger rogers" in name.casefold() for name in names)
    assert methods == {"bracket_match"}
    assert result["entities"][0]["in_kg"] is True


def test_real_2hop_bracket_question(kg: KnowledgeGraph, metaqa_dir: Path) -> None:
    question = _first_question(metaqa_dir / "2-hop" / "vanilla" / "qa_train.txt")
    assert "[Yuriy Norshteyn]" in question
    result = detect_entities(question, kg=kg)
    names = [item["name"] for item in result["entities"]]
    assert any("yuriy norshteyn" in name.casefold() for name in names)
    assert result["entities"][0]["method"] == "bracket_match"


def test_real_3hop_bracket_question(kg: KnowledgeGraph, metaqa_dir: Path) -> None:
    question = _first_question(metaqa_dir / "3-hop" / "vanilla" / "qa_train.txt")
    assert "[Dil Chahta Hai]" in question
    result = detect_entities(question, kg=kg)
    names = [item["name"] for item in result["entities"]]
    assert any("dil chahta hai" in name.casefold() for name in names)
    assert result["entities"][0]["method"] == "bracket_match"


def test_no_brackets_uses_index_match(kg: KnowledgeGraph) -> None:
    result = detect_entities("what movies are about ginger rogers", kg=kg)
    assert result["entities"], "expected at least one entity"
    assert result["entities"][0]["method"] == "index_match"
    assert "ginger rogers" in result["entities"][0]["name"].casefold()


def test_capitalization_differences(kg: KnowledgeGraph) -> None:
    bracketed = detect_entities("who directed [KISMET]", kg=kg)
    assert bracketed["entities"][0]["name"] == "Kismet"
    assert bracketed["entities"][0]["method"] == "bracket_match"

    unbracketed = detect_entities("who directed KISMET", kg=kg)
    names = [item["name"] for item in unbracketed["entities"]]
    assert "Kismet" in names
    assert unbracketed["entities"][0]["method"] == "index_match"


def test_multiple_entities(kg: KnowledgeGraph) -> None:
    result = detect_entities("compare [Kismet] and [The Sentinel]", kg=kg)
    names = [item["name"] for item in result["entities"]]
    assert "Kismet" in names
    assert "The Sentinel" in names
    assert len(result["entities"]) == 2
    assert all(item["method"] == "bracket_match" for item in result["entities"])

    unbracketed = detect_entities("compare Kismet and The Sentinel", kg=kg)
    unbracketed_names = [item["name"] for item in unbracketed["entities"]]
    assert "Kismet" in unbracketed_names
    assert "The Sentinel" in unbracketed_names
    assert all(item["method"] == "index_match" for item in unbracketed["entities"])


def test_unknown_entity_is_graceful(kg: KnowledgeGraph) -> None:
    unknown = "Totally Unknown Film XYZ 999"
    result = detect_entities(f"who directed [{unknown}]", kg=kg)
    assert len(result["entities"]) == 1
    assert result["entities"][0]["name"] == unknown
    assert result["entities"][0]["method"] == "bracket_match"
    assert result["entities"][0]["in_kg"] is False

    empty = detect_entities("who directed zzxqnotanentity999", kg=kg)
    assert empty["entities"] == []


def test_no_partial_word_matches(kg: KnowledgeGraph) -> None:
    """'artist' must not be detected inside 'artistically'."""
    result = detect_entities("what films can be described by artistically", kg=kg)
    names = [item["name"].casefold() for item in result["entities"]]
    assert "artist" not in names
