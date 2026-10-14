"""Load MetaQA kb.txt into a NetworkX MultiDiGraph with entity lookup."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

from backend.utils.normalize import extract_mention, normalize_entity, normalize_entity_name
from backend.utils.paths import find_kb_path


class KnowledgeGraph:
    """Directed multigraph of original MetaQA triples plus a normalized entity index."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        entity_index: dict[str, list[str]],
        malformed_count: int = 0,
        max_entity_tokens: int = 1,
    ) -> None:
        self.graph = graph
        self.entity_index = entity_index
        self.malformed_count = malformed_count
        self.max_entity_tokens = max_entity_tokens

    @classmethod
    def from_triples(
        cls,
        triples: list[tuple[str, str, str]],
        malformed_count: int = 0,
    ) -> "KnowledgeGraph":
        graph = nx.MultiDiGraph()
        for head, relation, tail in triples:
            if head not in graph:
                graph.add_node(head, original=head)
            if tail not in graph:
                graph.add_node(tail, original=tail)
            graph.add_edge(head, tail, relation=relation)

        entity_index: dict[str, list[str]] = defaultdict(list)
        for node in graph.nodes:
            key = normalize_entity(node)
            if node not in entity_index[key]:
                entity_index[key].append(node)
        index = dict(entity_index)
        max_entity_tokens = max((len(key.split()) for key in index), default=1)
        return cls(
            graph,
            index,
            malformed_count=malformed_count,
            max_entity_tokens=max_entity_tokens,
        )

    @classmethod
    def from_kb_file(cls, path: Path) -> "KnowledgeGraph":
        triples: list[tuple[str, str, str]] = []
        malformed = 0
        text = path.read_text(encoding="utf-8")
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                malformed += 1
                continue
            triples.append((parts[0], parts[1], parts[2]))
        return cls.from_triples(triples, malformed_count=malformed)

    def has_entity(self, original: str) -> bool:
        return original in self.graph

    def lookup_entity(self, mention: str) -> list[str]:
        key = normalize_entity(mention)
        return list(self.entity_index.get(key, []))

    def find_matching_entities(self, text: str) -> list[str]:
        return self.lookup_entity(extract_mention(text))

    def get_graph_statistics(self) -> dict:
        relations = [data["relation"] for _, _, data in self.graph.edges(data=True)]
        relation_counts = Counter(relations)
        return {
            "num_triples": self.graph.number_of_edges(),
            "num_nodes": self.graph.number_of_nodes(),
            "num_entities": self.graph.number_of_nodes(),
            "num_relations": len(relation_counts),
            "relation_counts": dict(relation_counts),
            "malformed_lines": self.malformed_count,
            "index_size": len(self.entity_index),
        }

    def statistics(self) -> dict:
        return self.get_graph_statistics()


def load_graph(kb_path: Path | None = None) -> KnowledgeGraph:
    path = Path(kb_path) if kb_path is not None else find_kb_path()
    return KnowledgeGraph.from_kb_file(path)


def get_graph_statistics(graph: KnowledgeGraph) -> dict:
    return graph.get_graph_statistics()


def find_matching_entities(graph: KnowledgeGraph, text: str) -> list[str]:
    return graph.find_matching_entities(text)
