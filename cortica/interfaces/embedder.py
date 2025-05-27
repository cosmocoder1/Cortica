"""Interfaces for embedding engines used in the Cortex memory system.

Defines the Embedder protocol, which specifies the required methods for
any embedding engine used to generate semantic vector representations of text.
"""

from typing import Protocol


class Embedder(Protocol):
    """Protocol for embedding engines that convert text into semantic vectors.

    Any object that implements this protocol can be used in the Cortex memory system
    to embed user messages for similarity-based recall.

    Required:
        - embed_query(text): Return a single vector for one string.
    Optional:
        - embed_documents(texts): Return a list of vectors for multiple strings.
    """

    def embed_query(self, text: str) -> list[float]:
        """Convert a single string into a vector embedding.

        Args:
            text (str): Input text to embed.

        Returns:
            List[float]: Semantic vector representation.
        """
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple strings into a list of vector embeddings.

        Args:
            texts (List[str]): List of input strings.

        Returns:
            List[List[float]]: Corresponding list of vector embeddings.
        """
        ...
