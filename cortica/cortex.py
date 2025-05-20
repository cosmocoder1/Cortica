"""
Cortex: High-level memory engine interface for semantic storage and retrieval.

This module acts as the public API for Cortica, allowing clients to:
- Store concept nodes with metadata or embeddings
- Query memory based on semantics, string keys, or context
- Integrate with external tools like LangChain, RAG pipelines, or custom NLP flows

Dependencies: None required to get started. Plug in embedding functions or storage layers as needed.
"""

from typing import Any, List, Dict, Optional
from .memory import MemoryGraph
from .embed import DefaultEmbedder


class Cortex:
    def __init__(self, embedder: Optional[Any] = None):
        """
        Initialize the Cortex engine.

        Args:
            embedder (Optional[Any]): An optional embedding function/class with `embed(text: str) -> List[float]`.
        """
        self.memory = MemoryGraph(use_decay=True, decay_half_life=3600)
        self.embedder = embedder or DefaultEmbedder()

    def remember(self, concept: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Store a concept node in memory.

        Args:
            concept (str): The core idea or content to store.
            metadata (dict, optional): Additional metadata (timestamp, source, etc.)
        """
        embedding = self.embedder.embed(concept)
        self.memory.store(concept, embedding, metadata)

    def query(self, prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant memories for a given prompt.

        Args:
            prompt (str): A search query or input string.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict[str, Any]]: Retrieved memory nodes with content and metadata.
        """
        embedding = self.embedder.embed(prompt)
        return self.memory.retrieve(embedding, top_k=top_k)
