"""
MemoryGraph: Core semantic memory store for Cortica.

This module manages storage and retrieval of concept nodes using vector similarity.
Designed to be lightweight and dependency-free unless extended.

Responsibilities:
- Store nodes with embeddings and optional metadata
- Retrieve top-k most similar nodes using cosine similarity
- Serve as the backend for Cortex-level memory operations
"""

from typing import List, Dict, Any
import math
from .decay import MemoryDecay


class MemoryGraph:
    def __init__(self, use_decay: bool = False, decay_half_life: float = 3600.0):
        self.nodes = []  # List of: {"concept": str, "embedding": List[float], "metadata": Dict}
        self.decay = MemoryDecay(half_life=decay_half_life) if use_decay else None

    def store(self, concept: str, embedding: List[float], metadata: Dict[str, Any] = None):
        """
        Store a new concept node in memory.

        Args:
            concept (str): The text content or idea to store.
            embedding (List[float]): Vector representation of the concept.
            metadata (dict, optional): Additional info (e.g., timestamp, tags)
        """
        self.nodes.append({
            "concept": concept,
            "embedding": embedding,
            "metadata": metadata or {}
        })

        if self.decay:
            self.decay.register(concept)

    def retrieve(self, query_embedding: List[float], top_k: int = 5, use_decay: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant memories, optionally weighted by freshness.

        Args:
            query_embedding (List[float]): Query vector.
            top_k (int): Number of top results.
            use_decay (bool): Whether to factor in decay strength.

        Returns:
            List[Dict[str, Any]]: Sorted memory nodes with content, metadata, and score.
        """
        scored = []
        for node in self.nodes:
            sim = self.cosine_similarity(query_embedding, node["embedding"])
            strength = 1.0

            if self.decay and use_decay:
                strength = self.decay.strength(node["concept"])

            score = sim * strength
            scored.append({**node, "score": score})

            if self.decay and use_decay:
                self.decay.register(node["concept"])  # Refresh memory on access

        top = sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]
        return top

    def prune(self, threshold: float = 0.1) -> int:
        """
        Forget memories below a strength threshold (if decay is enabled).

        Args:
            threshold (float): Strength cutoff.

        Returns:
            int: Number of memories removed.
        """
        if not self.decay:
            return 0

        retained = []
        removed = 0
        for node in self.nodes:
            if self.decay.should_forget(node["concept"], threshold=threshold):
                removed += 1
            else:
                retained.append(node)

        self.nodes = retained
        return removed

    def traverse(self, query_embedding: List[float], depth: int = 3) -> List[Dict[str, Any]]:
        """
        Traverse the memory graph starting from a query, chaining through similar concepts.

        Args:
            query_embedding (List[float]): Initial query vector.
            depth (int): Number of steps to walk through related memories.

        Returns:
            List[Dict[str, Any]]: Chain of related memory nodes.
        """
        path = []
        visited = set()

        current_embedding = query_embedding

        for _ in range(depth):
            candidates = []
            for node in self.nodes:
                if node["concept"] in visited:
                    continue

                sim = self.cosine_similarity(current_embedding, node["embedding"])
                candidates.append((sim, node))

            if not candidates:
                break

            candidates.sort(reverse=True, key=lambda x: x[0])
            top_node = candidates[0][1]
            path.append(top_node)
            visited.add(top_node["concept"])

            current_embedding = top_node["embedding"]

        return path

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-8)
