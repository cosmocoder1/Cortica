"""
MemoryGraph: Minimal vector-based memory store for Cortica.

This module manages storage and retrieval of user messages using embedding similarity.
It serves as the core backend for Cortex and supports:

- Storing message content with vector embeddings and optional metadata
- Retrieving the top-k most relevant memories via cosine similarity
- Providing a lightweight, dependency-free interface for short-term memory recall

No tagging, concept graphs, or NLP dependencies — designed for fast, semantic retrieval.
"""


import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass(frozen=True)
class MemoryNode:
    content: str
    vector: Tuple[float, ...]
    metadata: Dict[str, any] = field(default_factory=dict, compare=False)


class MemoryGraph:
    def __init__(self):
        self.memories: List[MemoryNode] = []

    def store(self, memory_text: str, embedding: List[float], metadata: Optional[Dict[str, any]] = None):
        memory = MemoryNode(
            content=memory_text,
            vector=tuple(embedding),
            metadata=metadata or {}
        )
        self.memories.append(memory)

    def retrieve(self, query_embedding: List[float], top_k: int = 5) -> List[MemoryNode]:
        scored = []
        for mem in self.memories:
            score = self.cosine_similarity(query_embedding, mem.vector)
            scored.append((score, mem))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [mem for _, mem in scored[:top_k]]

    @staticmethod
    def cosine_similarity(a: List[float], b: Tuple[float, ...]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-8)
