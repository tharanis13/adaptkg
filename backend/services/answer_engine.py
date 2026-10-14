"""Deterministic MetaQA answering from question wording and KG paths."""

from __future__ import annotations

import re
from typing import Any

from backend.services.entity_detection import BRACKET_RE, detect_entities
from backend.services.knowledge_graph import KnowledgeGraph, load_graph
from backend.utils.normalize import normalize_entity

REL_DIRECTED = "directed_by"
REL_WRITTEN = "written_by"
REL_STARRED = "starred_actors"
REL_GENRE = "has_genre"
REL_YEAR = "release_year"
REL_LANG = "in_language"
REL_TAGS = "has_tags"
REL_RATING = "has_imdb_rating"
REL_VOTES = "has_imdb_votes"

_SPACE = re.compile(r"\s+")


def _masked_question(question: str) -> str:
    masked = BRACKET_RE.sub(" ENT ", question)
    masked = re.sub(r"[,?]", " ", masked)
    return _SPACE.sub(" ", normalize_entity(masked)).strip()


def _has(q: str, *phrases: str) -> bool:
    return any(phrase in q for phrase in phrases)


def _hop(relation: str, direction: str, exclude_seeds: bool = False) -> dict[str, Any]:
    return {
        "relation": relation,
        "direction": direction,
        "exclude_seeds": exclude_seeds,
    }


def _tail_relation(q: str) -> str | None:
    if _has(q, "language"):
        return REL_LANG
    if _has(q, "genre", "genres", "what types", "what type"):
        return REL_GENRE
    if _has(q, "release", "released", "which years", "what year", "when did", "when were", "dates"):
        return REL_YEAR
    if _has(q, "screenwriter", "scriptwriter", "screenplay", "writer", "writers", "wrote", "written"):
        return REL_WRITTEN
    if _has(q, "director", "directors", "directed"):
        return REL_DIRECTED
    if _has(q, "actor", "actors", "acted", "starred", "star ", "stars ", "appear"):
        return REL_STARRED
    return None


def _share_bridge(q: str) -> str | None:
    if _has(
        q,
        "share actors",
        "share actor",
        "same actor",
        "same actors",
        "actors also appear",
        "actor of ent also",
        "actor in ent also",
        "starred by ent actors",
        "ent actors",
        "actors in ent",
        "actors in the film ent",
        "actors in the movie ent",
        "starred by the actors",
        "acted by the actors",
    ):
        return REL_STARRED
    if _has(
        q,
        "share directors",
        "share director",
        "same director",
        "directors also directed",
        "director also directed",
        "directed by the same director",
        "directed by the director of",
        "directed by the ent director",
        "ent director",
        "director of ent",
        "directors of their",
    ):
        return REL_DIRECTED
    if _has(
        q,
        "same screenwriter",
        "same writer",
        "screenwriter of ent also",
        "scriptwriter of ent also",
        "writer of ent",
        "written by the writer of",
        "writers of their",
    ):
        return REL_WRITTEN
    return None


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_SIMILARITY_THRESHOLD = 0.75

PROTOTYPE_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    # 1-hop
    "who directed ENT": [_hop(REL_DIRECTED, "forward")],
    "who is the director of ENT": [_hop(REL_DIRECTED, "forward")],
    "who directed the movie ENT": [_hop(REL_DIRECTED, "forward")],
    "who directed the film ENT": [_hop(REL_DIRECTED, "forward")],
    "ENT directed which movies": [_hop(REL_DIRECTED, "reverse")],
    "which movies did ENT direct": [_hop(REL_DIRECTED, "reverse")],
    "which films did ENT direct": [_hop(REL_DIRECTED, "reverse")],

    "who wrote ENT": [_hop(REL_WRITTEN, "forward")],
    "who is the writer of ENT": [_hop(REL_WRITTEN, "forward")],
    "who is the author of ENT": [_hop(REL_WRITTEN, "forward")],
    "who is the screenwriter of ENT": [_hop(REL_WRITTEN, "forward")],
    "who wrote the movie ENT": [_hop(REL_WRITTEN, "forward")],
    "who wrote the film ENT": [_hop(REL_WRITTEN, "forward")],
    "ENT wrote which movies": [_hop(REL_WRITTEN, "reverse")],
    "which movies did ENT write": [_hop(REL_WRITTEN, "reverse")],
    "which films did ENT write": [_hop(REL_WRITTEN, "reverse")],

    "who starred in ENT": [_hop(REL_STARRED, "forward")],
    "who acted in ENT": [_hop(REL_STARRED, "forward")],
    "who are the actors of ENT": [_hop(REL_STARRED, "forward")],
    "who are the actors in ENT": [_hop(REL_STARRED, "forward")],
    "the film ENT starred which actors": [_hop(REL_STARRED, "forward")],
    "the movie ENT starred which actors": [_hop(REL_STARRED, "forward")],
    "what does ENT appear in": [_hop(REL_STARRED, "reverse")],
    "what films does ENT appear in": [_hop(REL_STARRED, "reverse")],
    "what movies does ENT appear in": [_hop(REL_STARRED, "reverse")],
    "ENT appears in which movies": [_hop(REL_STARRED, "reverse")],
    "what films did ENT star in": [_hop(REL_STARRED, "reverse")],
    "what movies did ENT star in": [_hop(REL_STARRED, "reverse")],
    "what does ENT star in": [_hop(REL_STARRED, "reverse")],
    "what films did ENT act in": [_hop(REL_STARRED, "reverse")],

    "what is the genre of ENT": [_hop(REL_GENRE, "forward")],
    "what genre is ENT": [_hop(REL_GENRE, "forward")],
    "what kind of movie is ENT": [_hop(REL_GENRE, "forward")],
    "what genre of movie is ENT": [_hop(REL_GENRE, "forward")],
    "which movies are of genre ENT": [_hop(REL_GENRE, "reverse")],

    "when was ENT released": [_hop(REL_YEAR, "forward")],
    "what year was ENT released": [_hop(REL_YEAR, "forward")],
    "release year of ENT": [_hop(REL_YEAR, "forward")],
    "which movies were released in ENT": [_hop(REL_YEAR, "reverse")],

    "what is the language of ENT": [_hop(REL_LANG, "forward")],
    "what language is ENT": [_hop(REL_LANG, "forward")],
    "what languages are in ENT": [_hop(REL_LANG, "forward")],
    "which movies are in language ENT": [_hop(REL_LANG, "reverse")],

    "what is ENT about": [_hop(REL_TAGS, "forward")],
    "what words describe ENT": [_hop(REL_TAGS, "forward")],
    "can you give a few words describing ENT": [_hop(REL_TAGS, "forward")],
    "topics of ENT": [_hop(REL_TAGS, "forward")],
    "which movies are about ENT": [_hop(REL_TAGS, "reverse")],
    "what movies are about ENT": [_hop(REL_TAGS, "reverse")],

    "what is the imdb rating of ENT": [_hop(REL_RATING, "forward")],
    "what is the rating of ENT": [_hop(REL_RATING, "forward")],
    "what is the imdb votes of ENT": [_hop(REL_VOTES, "forward")],
    "how many votes does ENT have": [_hop(REL_VOTES, "forward")],

    # 2-hop
    "who directed the movies written by ENT": [_hop(REL_WRITTEN, "reverse"), _hop(REL_DIRECTED, "forward")],
    "who directed the movies starring ENT": [_hop(REL_STARRED, "reverse"), _hop(REL_DIRECTED, "forward")],
    "who wrote the movies directed by ENT": [_hop(REL_DIRECTED, "reverse"), _hop(REL_WRITTEN, "forward")],
    "who wrote the movies starring ENT": [_hop(REL_STARRED, "reverse"), _hop(REL_WRITTEN, "forward")],
    "who starred in the movies directed by ENT": [_hop(REL_DIRECTED, "reverse"), _hop(REL_STARRED, "forward")],
    "who starred in the movies written by ENT": [_hop(REL_WRITTEN, "reverse"), _hop(REL_STARRED, "forward")],
    "who are the actors in the films written by ENT": [_hop(REL_WRITTEN, "reverse"), _hop(REL_STARRED, "forward")],
    "who are the actors in the films directed by ENT": [_hop(REL_DIRECTED, "reverse"), _hop(REL_STARRED, "forward")],
    "which person directed the movies starred by ENT": [_hop(REL_STARRED, "reverse"), _hop(REL_DIRECTED, "forward")],
    "which person wrote the films directed by ENT": [_hop(REL_DIRECTED, "reverse"), _hop(REL_WRITTEN, "forward")],
    "the screenwriter ENT co wrote movies with who": [_hop(REL_WRITTEN, "reverse"), _hop(REL_WRITTEN, "forward", exclude_seeds=True)],
    "the movies starred by ENT were written by who": [_hop(REL_STARRED, "reverse"), _hop(REL_WRITTEN, "forward")],

    "what are the genres of the movies directed by ENT": [_hop(REL_DIRECTED, "reverse"), _hop(REL_GENRE, "forward")],
    "what are the genres of the movies written by ENT": [_hop(REL_WRITTEN, "reverse"), _hop(REL_GENRE, "forward")],
    "what are the genres of the movies starring ENT": [_hop(REL_STARRED, "reverse"), _hop(REL_GENRE, "forward")],
    "what are the genres of the films whose screenwriters also wrote ENT": [_hop(REL_WRITTEN, "forward"), _hop(REL_WRITTEN, "reverse", exclude_seeds=True), _hop(REL_GENRE, "forward")],

    "what are the primary languages in the movies directed by ENT": [_hop(REL_DIRECTED, "reverse"), _hop(REL_LANG, "forward")],
    "what are the languages of the movies directed by ENT": [_hop(REL_DIRECTED, "reverse"), _hop(REL_LANG, "forward")],
    "what are the languages of the movies written by ENT": [_hop(REL_WRITTEN, "reverse"), _hop(REL_LANG, "forward")],
    "what are the languages of the movies starring ENT": [_hop(REL_STARRED, "reverse"), _hop(REL_LANG, "forward")],
    "what are the languages spoken in the movies whose actors also appear in ENT": [_hop(REL_STARRED, "forward"), _hop(REL_STARRED, "reverse", exclude_seeds=True), _hop(REL_LANG, "forward")],

    "when were the movies directed by ENT released": [_hop(REL_DIRECTED, "reverse"), _hop(REL_YEAR, "forward")],
    "when were the movies written by ENT released": [_hop(REL_WRITTEN, "reverse"), _hop(REL_YEAR, "forward")],
    "when were the movies starring ENT released": [_hop(REL_STARRED, "reverse"), _hop(REL_YEAR, "forward")],

    "who starred in the same movie with ENT": [_hop(REL_STARRED, "reverse"), _hop(REL_STARRED, "forward", exclude_seeds=True)],
    "who appeared in the same movie with ENT": [_hop(REL_STARRED, "reverse"), _hop(REL_STARRED, "forward", exclude_seeds=True)],
    "who are movie co directors of ENT": [_hop(REL_DIRECTED, "reverse"), _hop(REL_DIRECTED, "forward", exclude_seeds=True)],
    "who are movie co writers of ENT": [_hop(REL_WRITTEN, "reverse"), _hop(REL_WRITTEN, "forward", exclude_seeds=True)],
    "which movies share the screenwriter with ENT": [_hop(REL_WRITTEN, "forward"), _hop(REL_WRITTEN, "reverse", exclude_seeds=True)],
    "which films share the screenwriter with ENT": [_hop(REL_WRITTEN, "forward"), _hop(REL_WRITTEN, "reverse", exclude_seeds=True)],

    # 3-hop
    "the films that share directors with the film ENT were in which languages": [_hop(REL_DIRECTED, "forward"), _hop(REL_DIRECTED, "reverse", exclude_seeds=True), _hop(REL_LANG, "forward")],
    "the films that share actors with the film ENT were in which languages": [_hop(REL_STARRED, "forward"), _hop(REL_STARRED, "reverse", exclude_seeds=True), _hop(REL_LANG, "forward")],
    "the films that share directors with the film ENT were in which genres": [_hop(REL_DIRECTED, "forward"), _hop(REL_DIRECTED, "reverse", exclude_seeds=True), _hop(REL_GENRE, "forward")],
    "the films that share actors with the film ENT were in which genres": [_hop(REL_STARRED, "forward"), _hop(REL_STARRED, "reverse", exclude_seeds=True), _hop(REL_GENRE, "forward")],
    "the films that share actors with the film ENT were released in which years": [_hop(REL_STARRED, "forward"), _hop(REL_STARRED, "reverse", exclude_seeds=True), _hop(REL_YEAR, "forward")],
    "the films that share directors with the film ENT were released in which years": [_hop(REL_DIRECTED, "forward"), _hop(REL_DIRECTED, "reverse", exclude_seeds=True), _hop(REL_YEAR, "forward")],
    "the films that share writers with the film ENT were released in which years": [_hop(REL_WRITTEN, "forward"), _hop(REL_WRITTEN, "reverse", exclude_seeds=True), _hop(REL_YEAR, "forward")],
    "what were the release years of the films that share writers with the film ENT": [_hop(REL_WRITTEN, "forward"), _hop(REL_WRITTEN, "reverse", exclude_seeds=True), _hop(REL_YEAR, "forward")],
    "who is listed as director of the films starred by ENT actors": [_hop(REL_STARRED, "reverse"), _hop(REL_STARRED, "forward", exclude_seeds=True), _hop(REL_DIRECTED, "forward")],
    "who starred movies for the director of ENT": [_hop(REL_DIRECTED, "reverse"), _hop(REL_DIRECTED, "forward"), _hop(REL_STARRED, "reverse")],
    "who directed movies for the writer of ENT": [_hop(REL_WRITTEN, "reverse"), _hop(REL_WRITTEN, "forward"), _hop(REL_DIRECTED, "reverse")],
    "who are the directors of the movies written by the writer of ENT": [_hop(REL_WRITTEN, "reverse"), _hop(REL_WRITTEN, "forward"), _hop(REL_DIRECTED, "reverse")],
    "what genres do the films that share directors with ENT fall under": [_hop(REL_DIRECTED, "forward"), _hop(REL_DIRECTED, "reverse", exclude_seeds=True), _hop(REL_GENRE, "forward")],
}


class TemplateMatcher:
    def __init__(self, templates: dict[str, list[dict[str, Any]]]):
        self.templates = list(templates.keys())
        self.plans = list(templates.values())
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.templates)
        
    def match(
        self, masked_question: str, max_hops: int, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
        # Filter template candidates matching target max_hops
        valid_indices = [i for i, p in enumerate(self.plans) if len(p) == max_hops]
        if not valid_indices:
            return None, {"matched_template": None, "similarity_score": 0.0, "used_fallback": True}

        valid_templates = [self.templates[i] for i in valid_indices]
        valid_plans = [self.plans[i] for i in valid_indices]

        q_vec = self.vectorizer.transform([masked_question])
        template_vecs = self.vectorizer.transform(valid_templates)
        scores = cosine_similarity(q_vec, template_vecs)[0]
        
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_template = valid_templates[best_idx]
        
        info = {
            "matched_template": best_template,
            "similarity_score": best_score,
            "used_fallback": best_score < threshold
        }
        
        if best_score >= threshold:
            return valid_plans[best_idx], info
        return None, info


_TEMPLATE_MATCHER = TemplateMatcher(PROTOTYPE_TEMPLATES)


def parse_hop_plan(
    question: str,
    max_hops: int,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[dict[str, Any]]:
    q = _masked_question(question)
    plan, info = _TEMPLATE_MATCHER.match(q, max_hops=max_hops, threshold=similarity_threshold)
    if plan is not None:
        return plan
        
    # Fallback to regex matching
    if max_hops == 1:
        return _plan_1hop(q)
    if max_hops == 2:
        return _plan_2hop(q)
    if max_hops == 3:
        return _plan_3hop(q)
    raise ValueError(f"max_hops must be 1, 2, or 3, got {max_hops}")



def _plan_1hop(q: str) -> list[dict[str, Any]]:
    movie_answer = _has(
        q,
        "which movie",
        "which film",
        "what movie",
        "what film",
        "what movies",
        "what films",
        "which movies",
        "which films",
        "what does ent appear",
        "what does ent star",
        "what does ent act",
        "what films did ent star",
        "what movies did ent star",
        "what films did ent act",
        "what movies did ent act",
        "what films does ent appear",
        "did ent direct",
        "did ent write",
        "did ent star",
        "did ent act",
        "does ent appear",
        "does ent star",
        "does ent act",
        "ent appears in",
        "ent directed which",
        "ent starred which",
        "was ent an actor",
        "was ent the writer",
        "was ent a writer",
        "written by ent",
        "movie written by",
        "film written by",
        "which films did ent direct",
        "which movies did ent direct",
        "which film did ent write",
        "which movie did ent write",
        "which movies was ent",
        "what movies was ent",
        "what films was ent",
        "what films did ent write",
        "what movies did ent write",
        "what film did ent write",
        "what movie did ent write",
    )
    if _has(q, "described by", "described with", "movies are about", "films are about", "topics is movie", "which topics"):
        if movie_answer or _has(q, "are about", "described by", "described with"):
            if _has(q, "topics is ent about", "topics is movie ent about", "what ent is about"):
                return [_hop(REL_TAGS, "forward")]
            return [_hop(REL_TAGS, "reverse")]
    if _has(q, "are about ent", "about ent") and movie_answer:
        return [_hop(REL_TAGS, "reverse")]
    if _has(q, "describe", "words", "terms", "topics") and not movie_answer:
        return [_hop(REL_TAGS, "forward")]
    if _has(q, "rating", "any good", "considered good", "think of", "audiences"):
        return [_hop(REL_RATING, "forward")]
    if _has(q, "popular", "famous", "popularity"):
        return [_hop(REL_VOTES, "forward")]
    if _has(q, "language"):
        return [_hop(REL_LANG, "reverse" if movie_answer else "forward")]
    if _has(q, "genre", "kind of", "sort of") or (_has(q, "type of movie", "type of film") and not movie_answer):
        return [_hop(REL_GENRE, "reverse" if movie_answer else "forward")]
    if _has(q, "release", "released", "what year", "when was", "when were"):
        return [_hop(REL_YEAR, "reverse" if movie_answer else "forward")]
    if _has(q, "direct"):
        return [_hop(REL_DIRECTED, "reverse" if movie_answer else "forward")]
    if _has(
        q,
        "write",
        "wrote",
        "writer",
        "written",
        "screenwriter",
        "screenplay",
        "script",
        "author",
        "story for",
    ):
        return [_hop(REL_WRITTEN, "reverse" if movie_answer else "forward")]
    if _has(q, "act", "star", "appear", "actor"):
        return [_hop(REL_STARRED, "reverse" if movie_answer else "forward")]
    return []


def _plan_2hop(q: str) -> list[dict[str, Any]]:
    if _has(q, "co-star", "co-stars", "together with", "same movie with", "acted together", "starred together"):
        return [_hop(REL_STARRED, "reverse"), _hop(REL_STARRED, "forward", exclude_seeds=True)]

    bridge = None
    if _has(q, "same actor", "share the same actor", "share same actor", "actor of ent also", "actor in ent also"):
        bridge = REL_STARRED
    elif _has(q, "same director", "share the same director", "share same director", "director of ent also", "director of ent is also", "directed by the same"):
        bridge = REL_DIRECTED
    elif _has(q, "same screenwriter", "screenwriter of ent also", "scriptwriter of ent also"):
        bridge = REL_WRITTEN
    if bridge:
        return [_hop(bridge, "forward"), _hop(bridge, "reverse", exclude_seeds=True)]

    person_rel = None
    if _has(q, "written by", "ent written", "films written", "movies written", "writer of", "wrote"):
        person_rel = REL_WRITTEN
    elif _has(q, "directed by", "films directed", "movies directed", "ent directed"):
        person_rel = REL_DIRECTED
    elif _has(q, "acted by", "starred by", "ent starred", "ent acted", "films starred", "movies starred", "movies acted", "films acted", "acted films", "starred movies"):
        person_rel = REL_STARRED
    elif _has(q, "director"):
        person_rel = REL_DIRECTED
    elif _has(q, "actor"):
        person_rel = REL_STARRED

    tail = _tail_relation(q)
    if person_rel and tail and tail != person_rel:
        return [_hop(person_rel, "reverse"), _hop(tail, "forward")]
    if person_rel and _has(q, "which movie", "which film", "what movie", "what film", "what are the movies", "what are the films"):
        return [_hop(person_rel, "reverse")]
    if person_rel and tail == person_rel:
        # e.g. who wrote films directed by X already set person_rel from "wrote" first
        other = None
        if person_rel != REL_DIRECTED and _has(q, "direct"):
            other = REL_DIRECTED
        elif person_rel != REL_WRITTEN and _has(q, "write", "wrote", "writer", "written"):
            other = REL_WRITTEN
        elif person_rel != REL_STARRED and _has(q, "act", "star", "appear"):
            other = REL_STARRED
        if other:
            return [_hop(other, "reverse"), _hop(person_rel, "forward")]
    if person_rel and tail:
        return [_hop(person_rel, "reverse"), _hop(tail, "forward")]
    return []


def _plan_3hop(q: str) -> list[dict[str, Any]]:
    bridge = _share_bridge(q)
    tail = _tail_relation(q)
    if bridge and tail:
        if tail == bridge:
            # Prefer a distinct tail if the same keyword family was reused.
            alt = None
            if bridge != REL_DIRECTED and _has(q, "direct"):
                alt = REL_DIRECTED
            elif bridge != REL_WRITTEN and _has(q, "write", "wrote", "writer", "written", "screenwriter"):
                alt = REL_WRITTEN
            elif bridge != REL_STARRED and _has(q, "starred who", "starred which", "which actors"):
                alt = REL_STARRED
            elif _has(q, "language"):
                alt = REL_LANG
            elif _has(q, "genre", "types"):
                alt = REL_GENRE
            elif _has(q, "year", "release", "when"):
                alt = REL_YEAR
            if alt:
                tail = alt
        return [
            _hop(bridge, "forward"),
            _hop(bridge, "reverse", exclude_seeds=True),
            _hop(tail, "forward"),
        ]
    return []


def _expand(
    kg: KnowledgeGraph,
    node: str,
    relation: str,
    direction: str,
) -> list[tuple[str, str, str, str]]:
    graph = kg.graph
    found: list[tuple[str, str, str, str]] = []
    if direction == "forward":
        edges = graph.out_edges(node, keys=True, data=True)
        for head, tail, _key, data in edges:
            if data["relation"] == relation:
                found.append((head, relation, tail, tail))
    else:
        edges = graph.in_edges(node, keys=True, data=True)
        for head, tail, _key, data in edges:
            if data["relation"] == relation:
                found.append((head, relation, tail, head))
    return found


def _walk(
    kg: KnowledgeGraph,
    seeds: list[str],
    hop_plan: list[dict[str, Any]],
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    seed_set = set(seeds)
    frontier: list[tuple[str, list[dict[str, Any]], set[str]]] = [
        (seed, [], {seed}) for seed in seeds
    ]
    for hop in hop_plan:
        nxt: dict[str, list[dict[str, Any]]] = {}
        path_nodes_for: dict[str, set[str]] = {}
        for node, path, seen in frontier:
            for head, relation, tail, neighbor in _expand(
                kg, node, hop["relation"], hop["direction"]
            ):
                if neighbor in seen:
                    continue
                if hop.get("exclude_seeds") and neighbor in seed_set:
                    continue
                if neighbor in nxt:
                    continue
                step = {
                    "subject": head,
                    "relation": relation,
                    "object": tail,
                    "direction": hop["direction"],
                    "depth": len(path) + 1,
                }
                nxt[neighbor] = path + [step]
                path_nodes_for[neighbor] = seen | {neighbor}
        frontier = [(n, p, path_nodes_for[n]) for n, p in nxt.items()]
        if not frontier:
            break
    answers = [node for node, _path, _seen in frontier]
    paths = [path for _node, path, _seen in frontier]
    return answers, paths


def answer_question(
    question: str,
    max_hops: int,
    kg: KnowledgeGraph | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict[str, Any]:
    graph = kg if kg is not None else load_graph()
    detected = detect_entities(question, kg=graph)
    seeds = [item["name"] for item in detected["entities"] if item.get("in_kg")]
    
    q = _masked_question(question)
    plan, info = _TEMPLATE_MATCHER.match(q, max_hops=max_hops, threshold=similarity_threshold)
    if plan is None:
        if max_hops == 1:
            plan = _plan_1hop(q)
        elif max_hops == 2:
            plan = _plan_2hop(q)
        elif max_hops == 3:
            plan = _plan_3hop(q)
        else:
            plan = []

    predicted, paths = _walk(graph, seeds, plan) if seeds and plan else ([], [])
    evidence = []
    for answer, path in zip(predicted, paths):
        evidence.append({"answer": answer, "path": path})
    return {
        "question": question,
        "detected_entities": detected["entities"],
        "seed_entities": seeds,
        "hop_plan": plan,
        "template_match_info": info,
        "predicted_answers": predicted,
        "evidence_paths": paths,
        "evidence": evidence,
        "answer_source": "kg_path",
        "max_hops": max_hops,
    }
