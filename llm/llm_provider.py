"""LLM access layer used by the agent."""

from __future__ import annotations

from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings


class LLMProvider:
    """Wrapper around OpenAI and Gemini with offline fallback."""

    def __init__(self) -> None:
        self._openai_client = (
            OpenAI(api_key=settings.openai_api_key, max_retries=0, timeout=8)
            if settings.use_openai and settings.openai_api_key
            else None
        )
        self._gemini_client = (
            ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                api_key=settings.google_api_key,
                temperature=0.2,
            )
            if settings.use_gemini and settings.google_api_key
            else None
        )

    def complete(self, prompt: str, *, temperature: float = 0.2) -> str:
        """Return a completion for the given prompt."""
        if self._gemini_client:
            try:
                response = self._gemini_client.invoke(prompt)
                return response.content.strip()
            except Exception:
                return self._offline_completion(prompt)
        elif self._openai_client:
            try:
                response = self._openai_client.chat.completions.create(
                    model=settings.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception:
                return self._offline_completion(prompt)

        return self._offline_completion(prompt)

    def _offline_completion(self, prompt: str) -> str:
        """Provide deterministic responses when no external LLM is configured."""
        lowered = prompt.lower()

        # Intent classification
        if "classify the user input" in lowered:
            user_text = lowered.split("user input:", 1)[-1]
            high_intent_terms = (
                "demo", "buy", "purchase", "sales", "start", "subscribe", 
                "trial", "contact", "sign up", "interested", "pro plan", 
                "upgrade", "i'd like to buy", "try the pro", "for my channel",
                "i want to start", "book a demo", "schedule a call"
            )
            greeting_terms = ("hello", "hi", "hey", "good morning", "good evening")
            if any(term in user_text for term in high_intent_terms):
                return "high_intent"
            if any(term in user_text for term in greeting_terms):
                return "greeting"
            return "product_query"

        # RAG query
        if "context:" in lowered and "question:" in lowered:
            context = prompt.split("Context:", 1)[-1].split("Question:", 1)[0].strip()
            question = prompt.split("Question:", 1)[-1].strip()
            return (
                "Based on AutoStream's knowledge base, "
                f"{self._answer_from_context(context, question)}"
            )

        return "I can help with AutoStream pricing, policies, and getting you connected with sales."

    def _answer_from_context(self, context: str, question: str) -> str:
        """Extract a concise grounded answer from retrieved context."""
        lowered_question = question.lower()
        relevant_lines = [
            line.strip("- ").strip()
            for line in context.splitlines()
            if line.strip() and any(token in line.lower() for token in lowered_question.split())
        ]
        if not relevant_lines:
            relevant_lines = [line.strip("- ").strip() for line in context.splitlines() if line.strip()]

        return " ".join(relevant_lines[:4])