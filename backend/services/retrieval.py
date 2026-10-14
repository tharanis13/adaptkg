"""Fixed-hop bidirectional retrieval over the MetaQA knowledge graph."""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from backend.services.entity_detection import detect_entities
from backend.services.knowledge_graph import KnowledgeGraph, load_graph

ALLOWED_HOPS = {1, 2, 3}


def _step_record(
    subject: str,
    relation: str,
    object_: str,
    depth: int,
    direction: str,
) -> dict[str, Any]:
    return {
        "subject": subject,
        "relation": relation,
        "object": object_,
        "depth": depth,
        "direction": direction,
    }


def _resolve_seeds(entities: list[str], kg: KnowledgeGraph) -> list[str]:
    seeds: list[str] = []
    seen: set[str] = set()
    for raw in entities:
        if not raw:
            continue
        names = [raw] if kg.has_entity(raw) else kg.lookup_entity(raw)
        for name in names:
            if kg.has_entity(name) and name not in seen:
                seen.add(name)
                seeds.append(name)
    return seeds


def _expand_edges(graph, node: str) -> list[tuple[str, str, str, str, str]]:
    """Return (head, relation, tail, neighbor, direction) for incident edges."""
    expansions: list[tuple[str, str, str, str, str]] = []
    for head, tail, _key, data in graph.out_edges(node, keys=True, data=True):
        expansions.append((head, data["relation"], tail, tail, "forward"))
    for head, tail, _key, data in graph.in_edges(node, keys=True, data=True):
        expansions.append((head, data["relation"], tail, head, "reverse"))
    return expansions


DEFAULT_MAX_CANDIDATES_PER_EXPANSION = 100
DEFAULT_MAX_TOTAL_CANDIDATES = 5000
DEFAULT_PRUNING_STRATEGY = "relevance_bounded"

RELATION_KEYWORDS = {
    "directed_by": {"director", "directed", "direct"},
    "written_by": {"writer", "written", "write", "screenwriter", "wrote", "author"},
    "starred_actors": {"actor", "actors", "star", "starred", "stars", "acted", "starring", "cast"},
    "has_genre": {"genre", "genres", "type", "category"},
    "release_year": {"year", "release", "released", "when", "date"},
    "in_language": {"language", "languages", "speak", "spoken", "english", "spanish", "french"},
    "has_tags": {"tag", "tags"},
    "has_imdb_rating": {"rating", "ratings", "score", "imdb"},
    "has_imdb_votes": {"votes", "vote", "imdb"},
}


def _score_expansion(
    relation: str,
    neighbor: str,
    question: str | None,
) -> float:
    if not question:
        return 0.0
    q_lower = question.lower()
    score = 0.0
    # Entity Match
    if len(neighbor) > 2 and neighbor.lower() in q_lower:
        score += 100.0
    # Relation Keyword Match
    keywords = RELATION_KEYWORDS.get(relation, set())
    for kw in keywords:
        if kw in q_lower:
            score += 1.0
    return score


def retrieve_fixed_hop(
    entities: list[str],
    max_hops: int,
    kg: KnowledgeGraph | None = None,
    *,
    question: str | None = None,
    max_candidates_per_expansion: int | None = DEFAULT_MAX_CANDIDATES_PER_EXPANSION,
    max_total_candidates: int | None = DEFAULT_MAX_TOTAL_CANDIDATES,
    pruning_strategy: str = DEFAULT_PRUNING_STRATEGY,
) -> dict[str, Any]:
    """Collect unique original triples within max_hops of the seed entities.

    Traversal is bidirectional. Reverse walks keep subject/relation/object as
    stored in kb.txt and set direction to "reverse".
    """
    if max_hops not in ALLOWED_HOPS:
        raise ValueError(f"max_hops must be one of {sorted(ALLOWED_HOPS)}, got {max_hops}")

    graph_wrapper = kg if kg is not None else load_graph()
    nx_graph = graph_wrapper.graph
    seeds = _resolve_seeds(list(entities), graph_wrapper)

    started = time.perf_counter()
    seen_triples: dict[tuple[str, str, str], dict[str, Any]] = {}
    visited: set[str] = set(seeds)
    queue: deque[tuple[str, int, list[str], list[dict[str, Any]]]] = deque(
        (seed, 0, [seed], []) for seed in seeds
    )

    while queue:
        # Check global limit
        if max_total_candidates is not None and len(seen_triples) >= max_total_candidates:
            break

        node, depth, path_nodes, path_steps = queue.popleft()
        if depth >= max_hops:
            continue

        expansions = _expand_edges(nx_graph, node)
        
        # Apply pruning strategy
        if pruning_strategy == "relevance_bounded" and max_candidates_per_expansion is not None:
            if len(expansions) > max_candidates_per_expansion:
                # Score each expansion
                scored_expansions = []
                for exp in expansions:
                    head, relation, tail, neighbor, direction = exp
                    score = _score_expansion(relation, neighbor, question)
                    scored_expansions.append((score, exp))
                
                # Sort descending by score, and ascending by (relation, neighbor, direction) for deterministic tie-breaker
                scored_expansions.sort(
                    key=lambda x: (-x[0], x[1][1], x[1][3], x[1][4])
                )
                
                # Slice to keep top-k
                expansions = [exp for score, exp in scored_expansions[:max_candidates_per_expansion]]

        for head, relation, tail, neighbor, direction in expansions:
            # Check global limit before adding
            if max_total_candidates is not None and len(seen_triples) >= max_total_candidates:
                break

            new_depth = depth + 1
            step = _step_record(head, relation, tail, new_depth, direction)
            new_path = path_steps + [step]
            key = (head, relation, tail)
            if key not in seen_triples:
                record = dict(step)
                record["path"] = new_path
                seen_triples[key] = record
            if neighbor in path_nodes:
                continue
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, new_depth, path_nodes + [neighbor], new_path))

    elapsed_ms = (time.perf_counter() - started) * 1000
    triples = list(seen_triples.values())
    return {
        "max_hops": max_hops,
        "seed_entities": seeds,
        "triples": triples,
        "num_triples": len(triples),
        "elapsed_ms": elapsed_ms,
    }


def retrieve_1hop(
    entities: list[str],
    kg: KnowledgeGraph | None = None,
    *,
    question: str | None = None,
    max_candidates_per_expansion: int | None = DEFAULT_MAX_CANDIDATES_PER_EXPANSION,
    max_total_candidates: int | None = DEFAULT_MAX_TOTAL_CANDIDATES,
    pruning_strategy: str = DEFAULT_PRUNING_STRATEGY,
) -> dict[str, Any]:
    return retrieve_fixed_hop(
        entities, 1, kg=kg, question=question,
        max_candidates_per_expansion=max_candidates_per_expansion,
        max_total_candidates=max_total_candidates,
        pruning_strategy=pruning_strategy
    )


def retrieve_2hop(
    entities: list[str],
    kg: KnowledgeGraph | None = None,
    *,
    question: str | None = None,
    max_candidates_per_expansion: int | None = DEFAULT_MAX_CANDIDATES_PER_EXPANSION,
    max_total_candidates: int | None = DEFAULT_MAX_TOTAL_CANDIDATES,
    pruning_strategy: str = DEFAULT_PRUNING_STRATEGY,
) -> dict[str, Any]:
    return retrieve_fixed_hop(
        entities, 2, kg=kg, question=question,
        max_candidates_per_expansion=max_candidates_per_expansion,
        max_total_candidates=max_total_candidates,
        pruning_strategy=pruning_strategy
    )


def retrieve_3hop(
    entities: list[str],
    kg: KnowledgeGraph | None = None,
    *,
    question: str | None = None,
    max_candidates_per_expansion: int | None = DEFAULT_MAX_CANDIDATES_PER_EXPANSION,
    max_total_candidates: int | None = DEFAULT_MAX_TOTAL_CANDIDATES,
    pruning_strategy: str = DEFAULT_PRUNING_STRATEGY,
) -> dict[str, Any]:
    return retrieve_fixed_hop(
        entities, 3, kg=kg, question=question,
        max_candidates_per_expansion=max_candidates_per_expansion,
        max_total_candidates=max_total_candidates,
        pruning_strategy=pruning_strategy
    )


def retrieve_for_question(
    question: str,
    max_hops: int,
    kg: KnowledgeGraph | None = None,
    *,
    max_candidates_per_expansion: int | None = DEFAULT_MAX_CANDIDATES_PER_EXPANSION,
    max_total_candidates: int | None = DEFAULT_MAX_TOTAL_CANDIDATES,
    pruning_strategy: str = DEFAULT_PRUNING_STRATEGY,
) -> dict[str, Any]:
    graph_wrapper = kg if kg is not None else load_graph()
    detected = detect_entities(question, kg=graph_wrapper)
    seed_names = [item["name"] for item in detected["entities"] if item.get("in_kg")]
    result = retrieve_fixed_hop(
        seed_names, max_hops, kg=graph_wrapper, question=question,
        max_candidates_per_expansion=max_candidates_per_expansion,
        max_total_candidates=max_total_candidates,
        pruning_strategy=pruning_strategy
    )
    result["question"] = question
    result["detected_entities"] = detected["entities"]
    return result

