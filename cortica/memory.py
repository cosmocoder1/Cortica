"""MemoryGraph: Minimal vector-based memory store for Cortica.

This module manages storage and retrieval of user messages using embedding similarity.
It serves as the core backend for Cortex and supports:

- Storing message content with vector embeddings and optional metadata
- Retrieving the top-k most relevant memories via cosine similarity
- Providing a lightweight, dependency-free interface for short-term memory recall

No tagging, concept graphs, or NLP dependencies — designed for fast, semantic retrieval.
"""


import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryNode:
    """Represents a single memory with content, embedding vector, and metadata.

    Used to store individual user messages or facts in the memory graph.
    Each node includes:
    - `content`: The original message or text snippet.
    - `vector`: A semantic embedding of the content for similarity retrieval.
    - `metadata`: Optional dictionary containing source info, tone, timestamp, etc.
    """
    content: str
    vector: tuple[float, ...]
    metadata: dict[str, any] = field(default_factory=dict, compare=False)


class MemoryGraph:
    """Lightweight memory store for user messages and their semantic embeddings.

    Manages short-term memory as a list of MemoryNodes, each containing message text,
    vector embeddings, and optional metadata (e.g., timestamp, tone). Supports appending
    new memories and retrieving the most relevant ones via vector similarity.
    Designed for fast, in-memory access in LLM context construction pipelines.
    """
    def __init__(self) -> None:
        self.memories: list[MemoryNode] = []

    def store(
        self,
        memory_text: str,
        embedding: list[float],
        metadata: dict[str, any] | None = None
    ) -> None:
        """Store a new memory node containing the user message and its embedding.

        Args:
            memory_text (str): Raw user message or input to store.
            embedding (list[float]): Vector representation of the message.
            metadata (dict[str, any] | None): Optional metadata (e.g., timestamp, source).
        """
        memory = MemoryNode(
            content=memory_text,
            vector=tuple(embedding),
            metadata=metadata or {}
        )
        self.memories.append(memory)

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[MemoryNode]:
        """Retrieve the top-k most relevant memories based on cosine similarity.

        Args:
            query_embedding (list[float]): Vector to compare against stored memories.
            top_k (int): Maximum number of results to return.

        Returns:
            list[MemoryNode]: Ranked list of memory nodes most similar to the query.
        """
        scored = []
        for mem in self.memories:
            score = self.cosine_similarity(query_embedding, mem.vector)
            scored.append((score, mem))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [mem for _, mem in scored[:top_k]]

    @staticmethod
    def cosine_similarity(a: list[float], b: tuple[float, ...]) -> float:
        """Compute the cosine similarity between two numeric vectors.

        Args:
            a (list[float]): First vector, typically the query embedding.
            b (tuple[float, ...]): Second vector, typically a stored memory embedding.

        Returns:
            float: Cosine similarity score between -1.0 and 1.0 (higher means more similar).

        Notes:
            - A small epsilon (1e-8) is added to the denominator to prevent division by zero.
            - Uses zip(strict=False) to gracefully handle any accidental length mismatch.
        """
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-8)

