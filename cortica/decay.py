"""
MemoryDecay: Time-aware memory weighting and retention logic for Cortica.

This module introduces lightweight aging mechanics to mimic cognitive memory decay:
- Memories lose strength over time unless refreshed
- Accessing a memory reinforces it
- Low-strength memories can be pruned or deprioritized
"""

import time
from typing import Dict, Optional


class MemoryDecay:
    def __init__(self, half_life: float = 3600.0):
        """
        Initialize the decay engine.

        Args:
            half_life (float): Time (in seconds) after which memory strength drops to 50%.
                               Default is 1 hour (3600 seconds).
        """
        self.half_life = half_life
        self.memory_times: Dict[str, float] = {}  # concept → last accessed timestamp

    def register(self, concept: str, timestamp: Optional[float] = None):
        """
        Register or update access time for a memory.

        Args:
            concept (str): The concept ID or name.
            timestamp (float, optional): Access time (defaults to now).
        """
        self.memory_times[concept] = timestamp or time.time()

    def strength(self, concept: str, now: Optional[float] = None) -> float:
        """
        Get the current strength of a memory based on decay.

        Args:
            concept (str): The concept ID or name.
            now (float, optional): Current time (defaults to time.time()).

        Returns:
            float: Normalized strength [0, 1].
        """
        now = now or time.time()
        last = self.memory_times.get(concept, now)
        delta = now - last
        return 0.5 ** (delta / self.half_life)

    def should_forget(self, concept: str, threshold: float = 0.1) -> bool:
        """
        Determine whether a memory should be forgotten based on strength.

        Args:
            concept (str): The concept ID or name.
            threshold (float): Minimum strength to retain.

        Returns:
            bool: True if the memory is weak enough to be discarded.
        """
        return self.strength(concept) < threshold
