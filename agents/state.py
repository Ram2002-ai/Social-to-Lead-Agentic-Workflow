"""State schema for the LangGraph workflow."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    messages: list[dict[str, str]]
    intent: str
    name: str | None
    email: str | None
    platform: str | None
    lead_collected: bool
    response: str