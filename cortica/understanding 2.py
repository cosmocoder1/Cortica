import os
import json
import re
from typing import Dict, Any
from pathlib import Path


class Understanding:
    """
    Extracts structured meaning from user input, such as name, location,
    interests, and other identity-related fields using lightweight pattern matching.
    """

    def __init__(self, mapping_path: str = None):
        """
        Initialize with a map of trigger phrases to identity fields.

        Args:
            mapping_path (str): Path to identity_map.json. Defaults to local mappings/ dir.
        """
        if mapping_path is None:
            mapping_path = Path(__file__).parent / "mappings" / "identity_map.json"

        with open(mapping_path, "r") as f:
            self.identity_map = json.load(f)

    def update_profile(self, text: str, profile: Dict[str, Any]):
        """
        Scan the text for identity-related fields and update the profile in place.

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
