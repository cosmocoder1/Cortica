"""This module provides Cortica with the ability to infer structured identity.

Details are gathered from natural language input. It scans user messages for semantic
cues such as name, location, interests, occupation, and more, using a trigger-based identity map.

Designed to enrich the user profile and generate a persistent sense of
relational continuity in LLM prompts.

Core Functionality:
- Load `identity_map.json` as the mapping layer
- Extract matching fields from raw text (e.g., "I'm from Austin" → location: "Austin")
- Update an in-memory user profile dictionary with new facts

Lightweight, transparent, and symbolic — no embeddings or NLP required.
"""

import json
import re
from pathlib import Path
from typing import Any


class Understanding:
    """Extracts structured meaning from user input.

    Establishes information such as name, location, interests, and other identity-related
    fields using lightweight pattern matching.
    """

    def __init__(self, identity_path: str = None, tone_path: str = None) -> None:
        """Load mappings from disk for identity fields and tone keywords."""
        base_path = Path(__file__).parent / "mappings"

        if identity_path is None:
            identity_path = base_path / "identity_map.json"
        if tone_path is None:
            tone_path = base_path / "tone_matrix.json"

        with open(identity_path) as f:
            self.identity_map = json.load(f)

        with open(tone_path) as f:
            self.tone_matrix = json.load(f)

    def update_profile(self, text: str, profile: dict[str, Any]) -> None:
        """Scan the text for identity-related fields and update the profile in place.

        Args:
            text (str): The raw user message.
            profile (Dict[str, Any]): The evolving user profile (dict-like).
        """
        lowered = text.lower()

        for field, phrases in self.identity_map.items():
            for phrase in phrases:
                if phrase in lowered:
                    # Pull everything after the phrase as a candidate value
                    match = re.search(re.escape(phrase) + r"\s+(.*)", lowered)
                    if match:
                        value = match.group(1).strip().rstrip(".!?")
                        if not value:
                            continue
                        # Append to list fields or store directly
                        if field in {"interests"}:
                            profile.setdefault(field, [])
                            if value not in profile[field]:
                                profile[field].append(value)
                        else:
                            if field not in profile:
                                profile[field] = value
                        break  # only process first matching phrase per field

    def infer_tone_vector(self, text: str) -> dict[str, float]:
        """Infer a 2D tone vector (valence, arousal) based on word presence.

        Args:
            text (str): The user's input sentence.

        Returns:
            Dict[str, float]: Normalized scores in the range [-1.0, 1.0] for:
                - 'valence'
                - 'arousal'
        """
        words = re.findall(r"\b\w+\b", text.lower())

        counts = {
            "valence": {"positive": 0, "negative": 0},
            "arousal": {"high": 0, "low": 0}
        }

        for word in words:
            for axis in ["valence", "arousal"]:
                for polarity in self.tone_matrix[axis]:
                    if word in self.tone_matrix[axis][polarity]:
                        counts[axis][polarity] += 1

        def normalize(pos: int, neg: int) -> float:
            total = pos + neg
            if total == 0:
                return 0.0
            return round((pos - neg) / total, 3)

        valence_score = normalize(counts["valence"]["positive"], counts["valence"]["negative"])
        arousal_score = normalize(counts["arousal"]["high"], counts["arousal"]["low"])

        return {
            "valence": valence_score,
            "arousal": arousal_score
        }