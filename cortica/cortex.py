"""
Cortex: Lightweight semantic memory engine for storing and retrieving past user inputs.

This module provides a simple API for:
- Storing user messages with vector embeddings
- Retrieving the most relevant prior messages given a new query
- Generating compressed context blocks to augment LLM prompts

Designed for use in chatbots, journaling apps, support agents, and RAG-style workflows.

Dependencies:
- Requires a user-supplied embedder with: embed_query(text: str) -> List[float]
- No NLP or tagging required. All memory is handled via vector similarity.
"""


import textwrap
from typing import Any, List, Dict
from cortica.memory import MemoryGraph


class Cortex:
    """
    Cortex is a minimal short-term memory module that stores user messages
    and retrieves relevant ones to provide compressed prompt context to an LLM.
    """

    def __init__(self, embedder: Any):
        """
        Initialize the Cortex engine with a required embedder.

        Args:
            embedder (Any): An object with `embed_query(text: str) -> List[float]`
        """
        if embedder is None:
            raise ValueError("Cortex requires an embedder. None was provided.")

        self.memory = MemoryGraph()
        self.embedder = embedder

    def remember(self, text: str):
        """
        Stores a user input and its embedding in memory.
        """
        embedding = self.embedder.embed_query(text)
        self.memory.store(
            memory_text=text,
            embedding=embedding,
            metadata={"source": "user", "content": text}
        )

    def recall(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant memories to the current query.
        """
        query_vector = self.embedder.embed_query(query)
        results = self.memory.retrieve(query_vector, top_k=top_k)
        return [r.metadata for r in results]

    def build_context_prompt(self, query: str, top_k: int = 5, width: int = 120, max_tokens: int = 800) -> str:
        """
        Build a context block for an LLM based on prior relevant user messages,
        constrained by a maximum token budget.

        Args:
            query (str): The user's current prompt.
            top_k (int): Maximum number of memories to consider.
            width (int): Character width limit per memory.
            max_tokens (int): Total token budget for the context block.

        Returns:
            str: LLM-ready prompt context string.
        """
        recalled = self.recall(query, top_k=top_k)

        context_lines = []
        used_tokens = 0

        for metadata in recalled:
            content = metadata.get("content", "[no content]")
            summary = textwrap.shorten(content, width=width, placeholder="...")
            token_count = self.estimate_tokens(summary)

            if used_tokens + token_count > max_tokens:
                break

            context_lines.append(f"- {summary}")
            used_tokens += token_count

        context_block = "\n".join(context_lines)
        return (
            "### Prior User Context\n"
            "The following entries summarize what the user has previously communicated. "
            "Use this context to respond in an informed, user-specific fashion.\n\n"
            f"{context_block}\n"
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count based on average token-to-character ratio (~1 token ≈ 4 characters).
        """
        return max(1, len(text) // 4)


