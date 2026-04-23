# AutoStream AI Sales Agent

A conversational AI agent for **AutoStream**, a fictional SaaS video editing platform. Built for the ServiceHive / Inflx ML Intern assignment.

---

## Features

- **RAG-Powered Knowledge Retrieval** - answers pricing, feature, and policy questions from a local JSON knowledge base
- **Intent Detection** - classifies user messages as casual, inquiry, or high_intent
- **Lead Qualification Flow** - collects name, email, and creator platform one field at a time
- **Tool Execution** - calls mock_lead_capture() only after all three fields are collected
- **State Management** - retains full conversation context across 5+ turns using LangGraph

---

## Project Structure

```
autostream-agent/
├── agent.py                        # Main agent logic (LangGraph + Groq Llama)
├── knowledge_base/
│   └── autostream_kb.json          # Local RAG knowledge base
├── requirements.txt
├── .env.example
└── README.md
```

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/autostream-agent.git
cd autostream-agent
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

Or export directly:

```bash
export GROQ_API_KEY=gsk_...
```

### 5. Run the agent

```bash
python agent.py
```

---

## Example Conversation

```
You: Hi! Tell me about your pricing.
Assistant: Hey! AutoStream has two plans: Basic at $29/month (10 videos, 720p) and Pro
          at $79/month (unlimited videos, 4K + AI captions). Which sounds right for you?

You: The Pro plan sounds perfect. I want to sign up for my YouTube channel.
Assistant: Awesome! I would love to get you set up. Could I get your name first?

You: Alex Johnson
Assistant: Nice to meet you, Alex! What is your email address?

You: alex@example.com
Assistant: Perfect! And which platform do you primarily create on?

You: YouTube
[Lead captured: Alex Johnson, alex@example.com, YouTube]
Assistant: You are all set, Alex! Our team will be in touch shortly. Welcome to AutoStream!
```

---

## Architecture Explanation

**Why LangGraph?**
LangGraph was chosen because it models the agent as an explicit state machine — each node in the graph has a clear responsibility (intent detection, field extraction, tool execution, response generation). This is ideal for a lead-capture flow where premature tool firing must be prevented. Unlike a simple ReAct loop, LangGraph allows deterministic conditional routing: the graph only routes to the tool node when all three required lead fields are present in state.

**How State is Managed**
A typed `AgentState` TypedDict tracks the full conversation history (via LangGraph's `add_messages` reducer), the detected intent, individual lead fields (name, email, platform), and boolean flags for whether collection is active and whether the lead has been captured. This state is passed through every graph node on each turn, so no information is lost between messages. The `add_messages` reducer appends new messages without overwriting history, giving the LLM full multi-turn context on every call.

**Graph Flow (per turn):**
```
User Input
    └─> intent_node   (detect/update intent, set collecting_lead flag)
    └─> extract_node  (parse next missing lead field from user message)
    └─> [conditional] tool_node (fire mock_lead_capture if all 3 fields present)
    └─> respond_node  (call Claude Haiku with full state-aware system prompt)
```

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, use the **WhatsApp Business API** (via Meta or a provider like Twilio) with a webhook architecture:

1. **Webhook Setup**: Register a public HTTPS endpoint (e.g. FastAPI on Railway or Render) as your WhatsApp webhook. Meta sends a POST request to this URL on every incoming message.

2. **Stateless Handler**: Since WhatsApp sends one message at a time, persist the `AgentState` in a database (Redis or DynamoDB) keyed by the sender phone number. On each webhook call: load state for that user, append the new message, run the LangGraph agent, save updated state, send the response back via the WhatsApp API.

3. **Response Delivery**: After the agent generates a reply, call the WhatsApp Send Message API with the AI response text.

4. **Lead Capture**: When `mock_lead_capture()` fires, replace it with a real CRM API call (HubSpot, Salesforce, etc.) or store in your database.

```
WhatsApp User
     |  (message)
     v
Meta Webhook → FastAPI Handler
                    |
                    ├── Load state from Redis (by phone number)
                    ├── Run LangGraph agent
                    ├── Save updated state to Redis
                    └── POST reply to WhatsApp Cloud API
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.9+ |
| Agent Framework | LangGraph 0.2+ |
| LLM | Llama 3.1 8B Instant (via Groq) |
| LLM SDK | LangChain Groq |
| Knowledge Base | Local JSON (RAG via prompt injection) |
| State Management | LangGraph StateGraph + TypedDict |
