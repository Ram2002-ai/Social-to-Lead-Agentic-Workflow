"""LLM-based intent classification."""

from __future__ import annotations

from llm.llm_provider import LLMProvider

VALID_INTENTS = {"greeting", "product_query", "high_intent"}


class IntentClassifier:
    """Classify user messages into the agent's supported intents."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def classify(self, user_input: str) -> str:
        """Classify input using the configured LLM provider."""
        prompt = f"""Classify the user input into one of:
- greeting
- product_query
- high_intent

Return ONLY the label.

User input: {user_input}
"""
        label = self.llm.complete(prompt, temperature=0).strip().lower()
        return label if label in VALID_INTENTS else "product_query"