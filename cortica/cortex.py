"""Cortex: Lightweight semantic memory engine for storing and retrieving past user inputs.

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
import time
from typing import Any

from cortica.memory import MemoryGraph
from cortica.profile import UserProfile
from cortica.understanding import Understanding


class Cortex:
    """Cortex is a minimal short-term memory module that stores user messages
    and retrieves relevant ones to provide compressed prompt context to an LLM.
    """

    def __init__(self, embedder: Any) -> None:
        """Initialize the Cortex engine with a required embedder.

        Args:
            embedder (Any): An object with `embed_query(text: str) -> List[float]`
        """
        if embedder is None:
            raise ValueError("Cortex requires an embedder. None was provided.")

        self.memory = MemoryGraph()
        self.embedder = embedder
        self.profile = UserProfile()
        self.understanding = Understanding()

    def remember(self, text: str):
        """Stores a user input and its embedding in memory.
        """
        embedding = self.embedder.embed_query(text)
        self.memory.store(
            memory_text=text,
            embedding=embedding,
            metadata={
                "source": "user",
                "content": text,
                "timestamp": time.time()
            }
        )

        # Infer identity and update profile
        self.understanding.update_profile(text, self.profile.fields)

    def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve the top-k most relevant memories to the current query.
        """
        query_vector = self.embedder.embed_query(query)
        results = self.memory.retrieve(query_vector, top_k=top_k)
        return [r.metadata for r in results]

    def build_context_prompt(self, query: str, top_k: int = 5, width: int = 120, max_tokens: int = 800) -> str:
        """Build a context block for an LLM based on prior relevant user messages,
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
        profile_summary = self.profile.summarize()
        now = time.time()

        for metadata in recalled:
            content = metadata.get("content", "[no content]")
            timestamp = metadata.get("timestamp")
            summary = textwrap.shorten(content, width=width, placeholder="...")
            token_count = self.estimate_tokens(summary)

            if used_tokens + token_count > max_tokens:
                break

            # Timestamp formatting
            if timestamp:
                elapsed_minutes = max(1, int((now - timestamp) / 60))
                time_note = f"({elapsed_minutes} min ago)"
            else:
                time_note = ""

            context_lines.append(f"- {summary} {time_note}")
            used_tokens += token_count

        context_block = "\n".join(context_lines)
        return (
            f"{profile_summary}\n\n"
            "### Prior User Context\n"
            "These are the latest communications they've sent you.\n"
            "Use this context to respond in an informed, user-specific fashion.\n\n"
            f"{context_block}\n"
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count based on average token-to-character ratio (~1 token ≈ 4 characters).
        """
        return max(1, len(text) // 4)
