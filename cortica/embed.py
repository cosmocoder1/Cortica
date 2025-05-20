"""
Embedding utilities for Cortica.

This module defines a default embedding interface for converting concepts into vector representations.
It supports:
- Dummy embedding for testing or offline use
- Optional OpenAI-based embedding for production-quality results
"""

from typing import List
import hashlib
import os

try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class DefaultEmbedder:
    def __init__(self, mode: str = "dummy"):
        """
        Initialize the embedder.

        Args:
            mode (str): Embedding mode — either "dummy" or "openai"
        """
        self.mode = mode

        if self.mode == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("langchain-openai not installed. Run `pip install langchain-openai`.")
            self.client = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    def embed(self, text: str) -> List[float]:
        """
        Generate an embedding for a given piece of text.

        Args:
            text (str): The input string.

        Returns:
            List[float]: A dense vector representing the input.
        """
        if self.mode == "openai":
            return self.client.embed_query(text)

        # Dummy mode: generate pseudo-embedding based on hash
        return self.dummy_embed(text)

    @staticmethod
    def dummy_embed(text: str, dim: int = 16) -> List[float]:
        """
        Generates a deterministic dummy embedding using a hash.

        Args:
            text (str): Input text.
            dim (int): Dimension of the vector.

        Returns:
            List[float]: Pseudo-embedding vector.
        """
        hash_bytes = hashlib.sha256(text.encode()).digest()
        return [b / 255 for b in hash_bytes[:dim]]
