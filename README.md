# AutoStream Conversational AI Agent

A production‑ready conversational agent for **AutoStream** (a fictional SaaS video editing platform). The agent identifies user intent, answers product questions using a local knowledge base (RAG), qualifies leads, and triggers a mock lead capture tool.

---

## 1. How to Run the Project Locally

### Prerequisites
- Python 3.9 or higher
- `pip` and `venv` (or `virtualenv`)

### Step‑by‑step

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/autostream-agent.git
cd autostream-agent

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Add API keys
# Copy the example environment file
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY or GOOGLE_API_KEY if desired.
# Without keys, the agent runs in offline mode (still fully functional).

# 5. Run the Streamlit app
streamlit run app.py



# 2. Architecture Explanation 
    Why LangGraph?
    LangGraph was chosen because it allows us to model the conversation as a state machine with cycles and conditional routing. Unlike simple LangChain chains, LangGraph gives fine‑grained control over the flow: the agent can detect intent, answer a RAG query, then later switch to lead collection without losing context. This is essential for a realistic sales agent that must pause mid‑flow, wait for user input, and resume exactly where it left off.

# How State is Managed
    The core state is defined in AgentState (a TypedDict) containing:

    messages: conversation history (role/content pairs)

    intent: current classification (greeting / product_query / high_intent)

    name, email, platform: lead data collected so far

    lead_collected: boolean flag indicating whether the mock API has been called

- Each node in the graph receives the current state, updates the relevant fields, and returns the new state. LangGraph passes this state automatically between nodes. Because the state is immutable (we return a new copy), debugging is straightforward and there are no unexpected side effects.

- For production persistence (e.g., across HTTP requests), LangGraph checkpointers can serialize the state to Redis or a database – but for this local demo, the state lives only in memory for the duration of the Streamlit session.

# 3. WhatsApp Integration Using Webhooks
- To integrate this agent with WhatsApp, follow this high‑level plan:

    Step 1 – WhatsApp Business API Access
    Obtain a WhatsApp Business API account (via Meta’s Cloud API or a provider like Twilio).
    This gives you a phone number, an access token, and a webhook verification endpoint.

    Step 2 – Deploy the Agent as a Web Service
    Wrap the agent in a lightweight web framework (e.g., FastAPI) and deploy it on a public URL (e.g., AWS, GCP, or Render).
- Expose two endpoints:

    GET /webhook – for Meta’s webhook verification (echoes a challenge token).

    POST /webhook – receives incoming WhatsApp messages.

    Step 3 – Webhook Payload Processing
    When a user sends a message to your WhatsApp number, Meta POSTs a JSON payload to your webhook.
    Extract:

    from – the user’s phone number (acts as session ID)

    text.body – the message content

    Step 4 – State Per User
    Maintain a dictionary or a Redis cache keyed by the phone number. Each entry holds the current AgentState.
    For each incoming message:

    Load the state for that phone number (or create a fresh one).

    Call agent.invoke(message) – the agent uses the loaded state as its starting point.

    Store the updated state back in the cache (with a TTL of e.g., 1 hour).

    Send the agent’s response back to WhatsApp via Meta’s messages endpoint.

    Step 5 – Reply via WhatsApp API
    Construct a POST request to:

    text
    https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages
    Include the recipient’s phone number, the message text, and your access token.

    Example Webhook Pseudocode
    python
    @app.post("/webhook")
    async def whatsapp_webhook(request: Request):
        data = await request.json()
        user_id = data["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
        user_text = data["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]

        state = load_from_redis(user_id) or AgentState(messages=[])
        agent = AutoStreamAgent(initial_state=state)
        response = agent.invoke(user_text)
        save_to_redis(user_id, agent.state)

        send_whatsapp_message(user_id, response)
        return {"status": "ok"}
    This approach makes the agent stateless at the server level while preserving conversation continuity for each WhatsApp user.