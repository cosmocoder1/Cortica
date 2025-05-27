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

from cortica.interfaces.embedder import Embedder
from cortica.memory import MemoryGraph
from cortica.profile import UserProfile
from cortica.understanding import Understanding


class Cortex:
    """Cortex is a short-term memory module that tracks and compresses user interaction."""

    def __init__(self, embedder: Embedder) -> None:
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

    def remember(self, text: str) -> None:
        """Stores a user input and its embedding in memory."""
        embedding = self.embedder.embed_query(text)

        # 🧠 Add tone inference
        tone_vector = self.understanding.infer_tone_vector(text)

        self.memory.store(
            memory_text=text,
            embedding=embedding,
            metadata={
                "source": "user",
                "content": text,
                "timestamp": time.time(),
                "tone_vector": tone_vector
            }
        )

    def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve the top-k most relevant memories to the current query."""
        query_vector = self.embedder.embed_query(query)
        results = self.memory.retrieve(query_vector, top_k=top_k)
        return [result.metadata for result in results]

    @staticmethod
    def average_tone(memories: list[dict[str, Any]]) -> tuple[float, float]:
        """Compute the average valence and arousal across a list of memory entries.

        Each memory is expected to include a 'tone_vector' with 'valence' and 'arousal' scores.
        Returns (0.0, 0.0) if no valid tone vectors are found.

        Args:
            memories (list[dict[str, Any]]): List of memory metadata dictionaries.

        Returns:
            tuple[float, float]: The average (valence, arousal) scores.
        """
        val_sum, ar_sum, count = 0.0, 0.0, 0
        for m in memories:
            tone = m.get("tone_vector")
            if tone:
                val_sum += tone.get("valence", 0.0)
                ar_sum += tone.get("arousal", 0.0)
                count += 1
        if count == 0:
            return (0.0, 0.0)

        return (round(val_sum / count, 3), round(ar_sum / count, 3))

    def build_context_prompt(
        self,
        query: str,
        top_k: int = 5,
        width: int = 120,
        max_tokens: int = 800
    ) -> str:
        """Build a compressed context block for an LLM using prior relevant user messages.

        Includes tone summaries over three time horizons and fits within a max token budget.

        Args:
            query (str): The user's current prompt.
            top_k (int): Number of memories to consider.
            width (int): Max width per memory snippet.
            max_tokens (int): Max total token budget.

        Returns:
            str: LLM-friendly context block.
        """
        recalled = self.recall(query, top_k=top_k)
        now = time.time()

        # Compute tone summaries for three time windows: short, medium, full
        def get_recent_tones(n: int) -> tuple[float, float]:
            return self.average_tone(recalled[-n:]) if len(recalled) >= n else (
                self.average_tone(recalled)
            )

        tone_short = get_recent_tones(2)
        tone_med = get_recent_tones(10)
        tone_full = self.average_tone(recalled)

        context_lines = []
        used_tokens = 0

        for metadata in recalled:
            content = metadata.get("content", "[no content]")
            summary = textwrap.shorten(content, width=width, placeholder="...")
            token_count = self.estimate_tokens(summary)

            if used_tokens + token_count > max_tokens:
                break

            minutes_ago = ""
            if ts := metadata.get("timestamp"):
                elapsed = max(1, int((now - ts) / 60))
                minutes_ago = f" ({elapsed} min ago)"

            # Only append summary and timing (no tone)
            context_lines.append(f"- {summary}{minutes_ago}")
            used_tokens += token_count

        context_block = "\n".join(context_lines)

        return (
            "You're speaking with a user.\n\n"
            "### Prior User Context\n"
            "These are the latest communications they've sent you.\n"
            "Use this context to respond in an informed, user-specific fashion.\n\n"
            f"{context_block}\n\n"
            "### Conversational Tone Summary (across time)\n"
            "These represent average emotional tone trends across three time horizons:\n"
            f"- **Last 2 messages**: valence={tone_short[0]:+.2f}, arousal={tone_short[1]:+.2f}\n"
            f"- **Last 10 messages**: valence={tone_med[0]:+.2f}, arousal={tone_med[1]:+.2f}\n"
            f"- **Overall**: valence={tone_full[0]:+.2f}, arousal={tone_full[1]:+.2f}\n"
            "Use these to understand how the user's tone may be shifting over time so "
            "that you can be more adaptive in your communication.\n"
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count based on average token-to-character ratio.

        Approximates how many tokens a given string would consume in an LLM prompt,
        assuming roughly 1 token per 4 characters (a common heuristic for English text).

        Args:
            text (str): The input text to analyze.

        Returns:
            int: Estimated number of tokens used by the input text.
        """
        return max(1, len(text) // 4)
