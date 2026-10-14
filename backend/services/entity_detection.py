"""Detect topic entities in MetaQA questions against the loaded KG."""

from __future__ import annotations

import re
from typing import Any

from backend.services.knowledge_graph import KnowledgeGraph, load_graph
from backend.utils.normalize import normalize_entity

BRACKET_RE = re.compile(r"\[([^\]]+)\]")
# Split a normalized question into tokens without breaking internal punctuation
# such as s.e. or 9/11.
_TOKEN_RE = re.compile(r"[^\s]+")
_EDGE_PUNCT = ".,;:!?\"'()[]{}"
METHOD_BRACKET = "bracket_match"
METHOD_INDEX = "index_match"


def _tokenize(text: str) -> list[str]:
    normalized = normalize_entity(text)
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(normalized):
        token = raw.strip(_EDGE_PUNCT)
        if token:
            tokens.append(token)
    return tokens


def _choose_canonical(mention: str, candidates: list[str]) -> str:
    if not candidates:
        return mention
    if mention in candidates:
        return mention
    casefold_hits = [c for c in candidates if c.casefold() == mention.casefold()]
    if casefold_hits:
        return casefold_hits[0]
    mixed_case = [c for c in candidates if any(ch.isupper() for ch in c)]
    if mixed_case:
        return mixed_case[0]
    return candidates[0]


def _entity_record(name: str, method: str, mention: str, in_kg: bool) -> dict[str, Any]:
    return {
        "name": name,
        "method": method,
        "mention": mention,
        "in_kg": in_kg,
    }


def _detect_from_brackets(question: str, kg: KnowledgeGraph) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in BRACKET_RE.finditer(question):
        mention = match.group(1).strip()
        if not mention:
            continue
        candidates = kg.lookup_entity(mention)
        if candidates:
            name = _choose_canonical(mention, candidates)
            in_kg = True
        else:
            name = mention
            in_kg = False
        key = (name, METHOD_BRACKET, mention)
        if key in seen:
            continue
        seen.add(key)
        entities.append(_entity_record(name, METHOD_BRACKET, mention, in_kg))
    return entities


def _detect_from_index(question: str, kg: KnowledgeGraph) -> list[dict[str, Any]]:
    tokens = _tokenize(question)
    if not tokens:
        return []

    max_n = max(getattr(kg, "max_entity_tokens", 1), 1)

    used = [False] * len(tokens)
    spans: list[tuple[int, int, str, list[str]]] = []

    for length in range(min(max_n, len(tokens)), 0, -1):
        for start in range(0, len(tokens) - length + 1):
            if any(used[start : start + length]):
                continue
            # Skip 1-character unigrams to avoid matching letters inside words
            # that became lone tokens after punctuation stripping.
            if length == 1 and len(tokens[start]) < 2:
                continue
            span = " ".join(tokens[start : start + length])
            candidates = kg.entity_index.get(span)
            if not candidates:
                continue
            for i in range(start, start + length):
                used[i] = True
            mention = " ".join(tokens[start : start + length])
            spans.append((start, start + length, mention, list(candidates)))

    spans.sort(key=lambda item: item[0])
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, _, mention, candidates in spans:
        name = _choose_canonical(mention, candidates)
        if name in seen:
            continue
        seen.add(name)
        entities.append(_entity_record(name, METHOD_INDEX, mention, True))
    return entities


def detect_entities(question: str, kg: KnowledgeGraph | None = None) -> dict[str, Any]:
    """Extract entities from a question.

    Bracketed mentions are used when present. Otherwise the question is scanned
    for exact normalized KG entity phrases (longest match first, token-bounded).
    """
    graph = kg if kg is not None else load_graph()
    stripped = question.strip()
    if BRACKET_RE.search(stripped):
        entities = _detect_from_brackets(stripped, graph)
    else:
        entities = _detect_from_index(stripped, graph)
    return {"entities": entities}
