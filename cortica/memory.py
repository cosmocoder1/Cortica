"""
MemoryGraph: Core semantic memory store for Cortica.

This module manages storage and retrieval of concept nodes using vector similarity.
Designed to be lightweight and dependency-free unless extended.

Responsibilities:
- Store nodes with embeddings and optional metadata
- Retrieve top-k most similar nodes using cosine similarity
- Serve as the backend for Cortex-level memory operations
"""

import math
from collections import defaultdict
from typing import List, Dict, Any, Optional

from cortica.decay import MemoryDecay


class MemoryGraph:
    def __init__(self, use_decay: bool = False, decay_half_life: float = 3600.0, link_threshold: float = 0.7):
        self.nodes: List[Dict[str, Any]] = []
        self.edges: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.decay = MemoryDecay(half_life=decay_half_life) if use_decay else None
        self.link_threshold = link_threshold

    def store(self, concept: str, embedding: List[float], metadata: Dict[str, Any] = None):
        self.nodes.append({
            "concept": concept,
            "embedding": embedding,
            "metadata": metadata or {}
        })

        if self.decay:
            self.decay.register(concept)

        # Auto-link with existing nodes
        for other in self.nodes[:-1]:  # skip the one we just added
            other_concept = other["concept"]
            sim = self.cosine_similarity(embedding, other["embedding"])
            if sim >= self.link_threshold:
                self.edges[concept][other_concept] = sim
                self.edges[other_concept][concept] = sim

    def retrieve(self, query_embedding: List[float], top_k: int = 5, use_decay: bool = True) -> List[Dict[str, Any]]:
        scored = []
        for node in self.nodes:
            sim = self.cosine_similarity(query_embedding, node["embedding"])
            strength = 1.0

            if self.decay and use_decay:
                strength = self.decay.strength(node["concept"])

            score = sim * strength
            scored.append({**node, "score": score})

            if self.decay and use_decay:
                self.decay.register(node["concept"])

        return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]

    def prune(self, threshold: float = 0.1) -> int:
        if not self.decay:
            return 0

        retained = []
        removed = 0
        removed_concepts = set()

        for node in self.nodes:
            concept = node["concept"]
            if self.decay.should_forget(concept, threshold=threshold):
                removed += 1
                removed_concepts.add(concept)
            else:
                retained.append(node)

        self.nodes = retained

        # Remove edges associated with pruned concepts
        for concept in removed_concepts:
            self.edges.pop(concept, None)
        for edge_dict in self.edges.values():
            for rc in removed_concepts:
                edge_dict.pop(rc, None)

        return removed

    def traverse(self, query_embedding: List[float], depth: int = 3) -> List[Dict[str, Any]]:
        path = []
        visited = set()

        # Find starting node
        start_node = max(
            self.nodes,
            key=lambda node: self.cosine_similarity(query_embedding, node["embedding"]),
            default=None
        )

        if not start_node:
            return path

        current = start_node["concept"]
        visited.add(current)
        path.append(start_node)

        for _ in range(depth - 1):
            neighbors = self.edges.get(current, {})
            next_node = None
            max_score = -1

            for neighbor, weight in neighbors.items():
                if neighbor in visited:
                    continue
                strength = self.decay.strength(neighbor) if self.decay else 1.0
                score = weight * strength
                if score > max_score:
                    max_score = score
                    next_node = neighbor

            if not next_node:
                break

            visited.add(next_node)
            current = next_node
            path.append(self.get_node_by_concept(current))

        return path

    def get_node_by_concept(self, concept: str) -> Optional[Dict[str, Any]]:
        for node in self.nodes:
            if node["concept"] == concept:
                return node
        return None

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-8)
