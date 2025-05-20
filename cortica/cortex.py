"""
Cortex: High-level memory engine interface for semantic storage and retrieval.

This module acts as the public API for Cortica, allowing clients to:
- Store concept nodes with metadata or embeddings
- Query memory based on semantics, string keys, or context
- Integrate with external tools like LangChain, RAG pipelines, or custom NLP flows

Dependencies: None required to get started. Plug in embedding functions or storage layers as needed.
"""

from typing import Any, List, Dict, Optional
from cortica.memory import MemoryGraph


class Cortex:
    """
    Cortex is a lightweight cognitive memory engine that stores and retrieves semantically embedded concepts.

    It requires an embedding backend to function. You must provide an embedder with a callable:
        embed(text: str) -> List[float]
    """

    def __init__(self, embedder: Any):
        """
        Initialize the Cortex engine with a required embedder.

        Args:
            embedder (Any): An embedding interface with `.embed(text: str) -> List[float]`
        """
        if embedder is None:
            raise ValueError("Cortex requires an embedder. None was provided.")

        self.memory = MemoryGraph(use_decay=True, decay_half_life=3600)
        self.embedder = embedder

    def remember(self, concept: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Store a concept node in memory.

        Args:
            concept (str): The idea or insight to store.
            metadata (dict, optional): Extra info such as timestamp, source, or tags.
        """
        embedding = self.embedder.embed_query(concept)
        self.memory.store(concept, embedding, metadata)

    def recall(self, prompt: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant memories to a query prompt.

        Args:
            prompt (str): A semantic query or question.
            top_k (int): Number of closest matches to return.

        Returns:
            List[Dict[str, Any]]: Ranked concepts with similarity scores and metadata.
        """
        embedding = self.embedder.embed_query(prompt)
        return self.memory.retrieve(embedding, top_k=top_k)
