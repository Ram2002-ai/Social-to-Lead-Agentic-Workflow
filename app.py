"""Responsive Streamlit frontend for the AutoStream agent."""

from __future__ import annotations

import streamlit as st

from agents.graph import AutoStreamAgent

st.set_page_config(
    page_title="AutoStream AI Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_agent() -> AutoStreamAgent:
    """Create or reuse the conversational agent for the current session."""
    if "agent" not in st.session_state:
        st.session_state.agent = AutoStreamAgent()
    return st.session_state.agent


def reset_chat() -> None:
    """Clear Streamlit and LangGraph conversation state."""
    st.session_state.agent = AutoStreamAgent()
    st.session_state.chat_messages = []


def render_styles() -> None:
    """Apply compact responsive styling to the Streamlit app."""
    st.markdown(
        """
        <style>
            .main .block-container {
                max-width: 1180px;
                padding-top: 1.4rem;
                padding-bottom: 6rem;
            }
            [data-testid="stSidebar"] {
                border-right: 1px solid #e5e7eb;
            }
            .hero {
                padding: 1.15rem 0 1.35rem;
                border-bottom: 1px solid #e5e7eb;
                margin-bottom: 1.2rem;
            }
            .hero h1 {
                font-size: 2.35rem;
                line-height: 1.08;
                margin: 0;
                color: #111827;
            }
            .hero p {
                max-width: 760px;
                margin: 0.55rem 0 0;
                color: #4b5563;
                font-size: 1rem;
            }
            .status-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.75rem;
                margin: 0 0 1.15rem;
            }
            .status-card {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 0.75rem 0.85rem;
                background: #ffffff;
            }
            .status-card span {
                display: block;
                color: #6b7280;
                font-size: 0.78rem;
                margin-bottom: 0.2rem;
            }
            .status-card strong {
                color: #111827;
                font-size: 0.95rem;
                overflow-wrap: anywhere;
            }
            @media (max-width: 760px) {
                .status-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
            @media (max-width: 430px) {
                .status-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def add_user_turn(prompt: str) -> None:
    """Send one message to the agent and record the response."""
    if not prompt.strip():
        return

    agent = get_agent()
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.spinner("AutoStream is thinking..."):
        response = agent.invoke(prompt)
    st.session_state.chat_messages.append({"role": "assistant", "content": response})


def render_status(agent: AutoStreamAgent) -> None:
    """Render live agent state for transparency during lead capture."""
    state = agent.state
    lead_status = "✓ Captured" if state.get("lead_collected") else "In progress"
    intent = state.get("intent") or "Waiting"

    st.markdown(
        f"""
        <div class="status-grid">
            <div class="status-card"><span>Intent</span><strong>{intent}</strong></div>
            <div class="status-card"><span>Name</span><strong>{state.get("name") or "Not collected"}</strong></div>
            <div class="status-card"><span>Email</span><strong>{state.get("email") or "Not collected"}</strong></div>
            <div class="status-card"><span>Lead</span><strong>{lead_status}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Run the Streamlit chat application."""
    render_styles()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    agent = get_agent()

    with st.sidebar:
        st.title("AutoStream")
        st.caption("Agentic RAG + Lead Capture")
        st.divider()
        st.write("Ask about pricing, support, refunds, or request a demo.")
        st.button("Reset conversation", use_container_width=True, on_click=reset_chat)
        st.divider()
        st.markdown("**Sample prompts**")
        if st.button("What is the Pro plan?", use_container_width=True):
            add_user_turn("What is the Pro plan?")
        if st.button("Do you offer refunds?", use_container_width=True):
            add_user_turn("Do you offer refunds?")
        if st.button("I want to book a demo", use_container_width=True):
            add_user_turn("I want to book a demo")

    st.markdown(
        """
        <section class="hero">
            <h1>AutoStream AI Agent</h1>
            <p>Ask product questions, compare plans, or start a sales conversation. The agent routes each turn through LangGraph, retrieves local product knowledge, and captures qualified leads only when the required details are ready.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_status(agent)

    if not st.session_state.chat_messages:
        with st.chat_message("assistant"):
            st.write("Hi! I can answer AutoStream product questions or help connect you with our team.")

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    prompt = st.chat_input("Ask AutoStream anything...")
    if prompt:
        add_user_turn(prompt)
        st.rerun()


if __name__ == "__main__":
    main()