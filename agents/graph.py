"""LangGraph workflow definition and agent facade."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.nodes import AgentNodes
from agents.state import AgentState
from llm.llm_provider import LLMProvider
from rag.retriever import KnowledgeBaseRetriever
from utils.intent_classifier import IntentClassifier


class AutoStreamAgent:
    """Stateful CLI-friendly facade over the LangGraph workflow."""

    def __init__(self) -> None:
        self.state: AgentState = {
            "messages": [],
            "intent": "",
            "name": None,
            "email": None,
            "platform": None,
            "lead_collected": False,
            "response": "",
        }
        self.llm = LLMProvider()
        self.retriever = KnowledgeBaseRetriever()
        self.classifier = IntentClassifier(self.llm)
        self.nodes = AgentNodes(self.llm, self.retriever, self.classifier)
        self.graph = self._build_graph()

    def invoke(self, user_input: str) -> str:
        """Process one user turn and return the agent response."""
        self.state["messages"].append({"role": "user", "content": user_input})
        self.state = self.graph.invoke(self.state)
        return self.state["response"]

    def _build_graph(self):
        """Create and compile the LangGraph state machine."""
        graph = StateGraph(AgentState)

        graph.add_node("detect_intent", self.nodes.detect_intent)
        graph.add_node("handle_greeting", self.nodes.handle_greeting)
        graph.add_node("handle_rag_query", self.nodes.handle_rag_query)
        graph.add_node("handle_lead_flow", self.nodes.handle_lead_flow)

        graph.set_entry_point("detect_intent")
        graph.add_conditional_edges(
            "detect_intent",
            self.nodes.route_intent,
            {
                "handle_greeting": "handle_greeting",
                "handle_rag_query": "handle_rag_query",
                "handle_lead_flow": "handle_lead_flow",
            },
        )

        graph.add_edge("handle_greeting", END)
        graph.add_edge("handle_rag_query", END)
        graph.add_edge("handle_lead_flow", END)

        return graph.compile()