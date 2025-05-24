"""Defines the UserProfile class.

This stores structured identity information extracted from user input. This profile
provides symbolic continuity to an otherwise stateless LLM, enabling Cortica to simulate
relational memory.

Responsibilities:
- Store known user facts (name, location, interests, etc.)
- Merge new inferred data without duplication
- Summarize the profile in natural language for prompt context

This class is designed to be dict-like, lightweight, and easily extensible.
"""

from typing import Any


class UserProfile:
    """Represents a symbolic identity profile for the user.

    Includes name, location, interests, and other personal facts.
    """

    def __init__(self) -> None:
        self.fields: dict[str, Any] = {}

    def update(self, new_data: dict[str, Any]) -> None:
        """Merge new identity fields into the profile.

        Args:
            new_data (Dict[str, Any]): A dictionary of new facts, e.g. {"location": "Austin"}
        """
        for key, value in new_data.items():
            if key in {"interests"}:
                self.fields.setdefault(key, [])
                if isinstance(value, list):
                    for item in value:
                        if item not in self.fields[key]:
                            self.fields[key].append(item)
                else:
                    if value not in self.fields[key]:
                        self.fields[key].append(value)
            else:
                if key not in self.fields:
                    self.fields[key] = value

    def summarize(self) -> str:
        """Generate a natural language summary of the profile for use in prompts.

        Returns:
            str: A short description like "You're speaking with Alex from Austin who enjoys jazz."
        """
        summary_parts = []

        name = self.fields.get("name")
        if name:
            summary_parts.append(f"You're speaking with {name}")
        else:
            summary_parts.append("You're speaking with a user")

        location = self.fields.get("location")
        if location:
            summary_parts[-1] += f" from {location}"

        occupation = self.fields.get("occupation")
        if occupation:
            summary_parts.append(f"who works as a {occupation}")

        interests = self.fields.get("interests", [])
        if interests:
            interests_str = ", ".join(interests[:3])
            summary_parts.append(f"and enjoys {interests_str}")

        return " ".join(summary_parts) + "."

    def to_dict(self) -> dict[str, Any]:
        """Return the profile fields as a dictionary."""
        return self.fields.copy()
