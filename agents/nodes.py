"""LangGraph node implementations for the AutoStream agent."""

from __future__ import annotations

import re
from dataclasses import dataclass

from agents.state import AgentState
from llm.llm_provider import LLMProvider
from rag.retriever import KnowledgeBaseRetriever
from tools.lead_capture import mock_lead_capture
from utils.intent_classifier import IntentClassifier

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass
class AgentNodes:
    """Container for workflow node functions and shared dependencies."""

    llm: LLMProvider
    retriever: KnowledgeBaseRetriever
    classifier: IntentClassifier

    def detect_intent(self, state: AgentState) -> AgentState:
        """Classify the latest user message and update state."""
        user_input = state["messages"][-1]["content"]
        # If we're already in lead flow, force high_intent
        if self._lead_in_progress(state):
            state["intent"] = "high_intent"
            return state

        state["intent"] = self.classifier.classify(user_input)
        return state

    def handle_greeting(self, state: AgentState) -> AgentState:
        """Return a friendly greeting response."""
        state["response"] = (
            "Hi! I can help with AutoStream pricing, policies, features, "
            "or connect you with our team for a demo."
        )
        self._append_assistant_message(state)
        return state

    def handle_rag_query(self, state: AgentState) -> AgentState:
        """Retrieve local context and answer the user's product question."""
        question = state["messages"][-1]["content"]
        context = self.retriever.context_for(question)
        prompt = f"""Answer the user's AutoStream question using only this context.
If the answer is not in the context, say you do not have that information.

Context:
{context}

Question: {question}
"""
        state["response"] = self.llm.complete(prompt)
        self._append_assistant_message(state)
        return state

    def handle_lead_flow(self, state: AgentState) -> AgentState:
        """Start or continue the lead qualification flow."""
        if state.get("lead_collected"):
            state["response"] = (
                "Your details are already captured. Our team will follow up with you soon."
            )
            self._append_assistant_message(state)
            return state

        if not state.get("name"):
            return self.collect_name(state)
        if not state.get("email"):
            return self.collect_email(state)
        if not state.get("platform"):
            return self.collect_platform(state)
        return self.execute_tool(state)

    def collect_name(self, state: AgentState) -> AgentState:
        """Collect the lead's name when missing."""
        latest = state["messages"][-1]["content"].strip()
        if self._awaiting_field(state, "name") and self._looks_like_name(latest):
            state["name"] = latest
            state["response"] = "Thanks. What email should our team use to contact you?"
        else:
            state["response"] = "I can help set that up. What is your name?"
        self._append_assistant_message(state)
        return state

    def collect_email(self, state: AgentState) -> AgentState:
        """Collect and validate the lead's email address."""
        latest = state["messages"][-1]["content"].strip()
        if EMAIL_PATTERN.match(latest):
            state["email"] = latest
            state["response"] = "Great. Which platform do you plan to use AutoStream on? (YouTube, Instagram, TikTok, etc.)"
        else:
            state["response"] = "Please share a valid email address so our team can reach you."
        self._append_assistant_message(state)
        return state

    def collect_platform(self, state: AgentState) -> AgentState:
        """Collect the lead's intended publishing platform."""
        latest = state["messages"][-1]["content"].strip()
        if len(latest) >= 2:
            state["platform"] = latest
            return self.execute_tool(state)

        state["response"] = "Which platform will you use AutoStream for, such as YouTube, Instagram, or TikTok?"
        self._append_assistant_message(state)
        return state

    def execute_tool(self, state: AgentState) -> AgentState:
        """Execute lead capture only after all required fields exist."""
        name = state.get("name")
        email = state.get("email")
        platform = state.get("platform")
        if not (name and email and platform):
            return self.handle_lead_flow(state)

        mock_lead_capture(name, email, platform)
        state["lead_collected"] = True
        state["response"] = (
            f"Thanks, {name}. Our team will contact you at {email} about using "
            f"AutoStream for {platform}. We'll reach out within 24 hours."
        )
        self._append_assistant_message(state)
        return state

    def route_intent(self, state: AgentState) -> str:
        """Route the workflow according to detected intent."""
        intent = state.get("intent", "product_query")
        if intent == "greeting":
            return "handle_greeting"
        if intent == "high_intent":
            return "handle_lead_flow"
        return "handle_rag_query"

    def _append_assistant_message(self, state: AgentState) -> None:
        """Add the node response to message history."""
        state["messages"].append({"role": "assistant", "content": state["response"]})

    def _lead_in_progress(self, state: AgentState) -> bool:
        """Return true when a lead flow has started but not completed."""
        has_partial_lead = bool(state.get("name") or state.get("email") or state.get("platform"))
        awaiting_lead_detail = any(
            self._awaiting_field(state, field)
            for field in ("name", "email", "platform")
        )
        return (has_partial_lead or awaiting_lead_detail) and not state.get("lead_collected")

    def _looks_like_name(self, value: str) -> bool:
        """Check whether a short user response can reasonably be a name."""
        if EMAIL_PATTERN.match(value):
            return False
        return 1 <= len(value.split()) <= 4 and len(value) >= 2

    def _awaiting_field(self, state: AgentState, field_name: str) -> bool:
        """Infer which lead field was requested in the previous assistant turn."""
        if len(state["messages"]) < 2:
            return False

        previous_message = state["messages"][-2]
        if previous_message.get("role") != "assistant":
            return False

        content = previous_message.get("content", "").lower()
        expected_phrases = {
            "name": ("what is your name",),
            "email": ("what email", "valid email"),
            "platform": ("which platform",),
        }
        return any(phrase in content for phrase in expected_phrases[field_name])