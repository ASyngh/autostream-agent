"""
AutoStream Conversational AI Agent
===================================
Built with LangGraph for state management and RAG-powered knowledge retrieval.
Compatible with Python 3.9+.
"""

from __future__ import annotations

import json
import os
import re
from typing import Annotated, TypedDict, Literal
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ─────────────────────────────────────────────
# 1. KNOWLEDGE BASE (RAG)
# ─────────────────────────────────────────────

def load_knowledge_base() -> str:
    """Load the local JSON knowledge base and convert to a readable string for the LLM."""
    kb_path = Path(__file__).parent / "knowledge_base" / "autostream_kb.json"
    with open(kb_path, "r") as f:
        kb = json.load(f)

    kb_text = f"""
AUTOSTREAM KNOWLEDGE BASE
==========================

COMPANY:
{kb['company']['name']} — {kb['company']['description']}
Tagline: {kb['company']['tagline']}

PRICING PLANS:
"""
    for plan in kb["pricing"]["plans"]:
        kb_text += f"\n{plan['name']} — ${plan['price_monthly']}/month\n"
        kb_text += f"  Best for: {plan['best_for']}\n"
        kb_text += "  Features:\n"
        for feat in plan["features"]:
            kb_text += f"    • {feat}\n"

    kb_text += f"""
COMPANY POLICIES:
- Refund Policy: {kb['policies']['refund_policy']}
- Basic Support: {kb['policies']['support']['basic']}
- Pro Support: {kb['policies']['support']['pro']}
- Trial: {kb['policies']['trial']}
- Cancellation: {kb['policies']['cancellation']}

FAQS:
"""
    for faq in kb["faqs"]:
        kb_text += f"Q: {faq['question']}\nA: {faq['answer']}\n\n"

    return kb_text


KNOWLEDGE_BASE = load_knowledge_base()


# ─────────────────────────────────────────────
# 2. MOCK LEAD CAPTURE TOOL
# ─────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock API function to capture a qualified lead."""
    print(f"\n{'='*50}")
    print(f"✅ Lead captured successfully: {name}, {email}, {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"


# ─────────────────────────────────────────────
# 3. AGENT STATE
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str                  # casual | inquiry | high_intent
    lead_name: str | None
    lead_email: str | None
    lead_platform: str | None
    lead_captured: bool
    collecting_lead: bool        # flag: currently in lead collection flow


# ─────────────────────────────────────────────
# 4. LLM SETUP
# ─────────────────────────────────────────────

def get_llm():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.3,
        max_tokens=1024,
    )


# ─────────────────────────────────────────────
# 5. SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────

def build_system_prompt(state: AgentState) -> str:
    lead_status = ""
    if state.get("collecting_lead"):
        collected = []
        missing = []
        for field, val in [("name", state.get("lead_name")),
                            ("email", state.get("lead_email")),
                            ("platform", state.get("lead_platform"))]:
            if val:
                collected.append(f"{field}: {val}")
            else:
                missing.append(field)
        lead_status = f"""
LEAD COLLECTION IN PROGRESS: 
  Collected so far: {', '.join(collected) if collected else 'nothing yet'}
  Still needed: {', '.join(missing)}
  
Ask ONLY for the next missing field. Do NOT ask for multiple fields at once.
Once all three are collected, confirm the details and say you are submitting them.
"""

    return f"""You are an AI sales assistant for AutoStream, a SaaS video editing platform for content creators.

Your personality: Friendly, knowledgeable, concise. You help creators understand the product and guide high-intent users toward signing up.

KNOWLEDGE BASE (use this to answer all product questions accurately):
{KNOWLEDGE_BASE}

CURRENT CONVERSATION STATE:
- Detected intent: {state.get('intent', 'unknown')}
- Lead collection active: {state.get('collecting_lead', False)}
- Lead already captured: {state.get('lead_captured', False)}
{lead_status}

BEHAVIORAL RULES:
1. If the user asks about pricing, features, or policies — answer using the knowledge base only.
2. If you detect HIGH INTENT (user says things like "I want to sign up", "I'm ready", "I want to try the Pro plan", "sign me up", "let's do it") — begin collecting their details one field at a time.
3. Collect fields in this order: Name → Email → Creator Platform (YouTube, Instagram, TikTok, etc.)
4. Do NOT ask for all fields at once. Ask one at a time.
5. Never make up information not in the knowledge base.
6. Keep responses concise and conversational — 2–4 sentences max unless explaining pricing.
7. If lead is already captured, thank the user and offer further help.
"""


# ─────────────────────────────────────────────
# 6. INTENT DETECTION
# ─────────────────────────────────────────────

HIGH_INTENT_PHRASES = [
    r"\bsign\s*me\s*up\b",
    r"\bi('m| am)\s*(ready|interested)\b",
    r"\bwant\s*to\s*(try|subscribe|get|buy|start|join|sign up)\b",
    r"\blet'?s\s*do\s*(it|this)\b",
    r"\bwhere\s*do\s*i\s*(sign|pay|subscribe)\b",
    r"\bpurchase\b",
    r"\bi('ll| will)\s*take\b",
    r"\btake\s*the\s*(pro|basic)\s*plan\b",
    r"\bget\s*started\b",
    r"\bonboard\b",
]

GREETING_PHRASES = [
    r"^\s*(hi|hello|hey|howdy|good\s(morning|evening|afternoon)|what'?s\s*up)\b"
]


def detect_intent(user_message: str, current_intent: str) -> str:
    """Rule-based intent detection with carry-forward logic."""
    msg = user_message.lower()

    # If already in high-intent / collecting, don't downgrade
    if current_intent == "high_intent":
        return "high_intent"

    for pattern in HIGH_INTENT_PHRASES:
        if re.search(pattern, msg):
            return "high_intent"

    for pattern in GREETING_PHRASES:
        if re.search(pattern, msg):
            return "casual"

    # Keywords that suggest inquiry
    inquiry_keywords = ["price", "plan", "cost", "feature", "refund", "support",
                        "resolution", "video", "caption", "4k", "720", "pro", "basic",
                        "cancel", "trial", "policy", "upgrade", "how"]
    if any(kw in msg for kw in inquiry_keywords):
        return "inquiry"

    return current_intent or "casual"


# ─────────────────────────────────────────────
# 7. LEAD FIELD EXTRACTION
# ─────────────────────────────────────────────

def extract_email(text: str) -> str | None:
    match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None


PLATFORMS = ["youtube", "instagram", "tiktok", "twitter", "facebook", "twitch",
             "linkedin", "snapchat", "x", "shorts"]


def extract_platform(text: str) -> str | None:
    lower = text.lower()
    for p in PLATFORMS:
        if p in lower:
            return p.capitalize()
    return None


def extract_lead_fields(state: AgentState, user_message: str) -> AgentState:
    """
    Progressively extract lead fields from the conversation.
    Only extract the next missing field to avoid premature extraction.
    """
    updated = dict(state)

    if not updated.get("lead_name"):
        # Heuristic: name is likely if message is 1-3 words with no special chars, no email
        words = user_message.strip().split()
        email_in_msg = extract_email(user_message)
        platform_in_msg = extract_platform(user_message)
        if (1 <= len(words) <= 4
                and not email_in_msg
                and not platform_in_msg
                and not any(c in user_message for c in ["?", "@", "http"])
                and user_message[0].isupper()):
            updated["lead_name"] = user_message.strip()
        return updated

    if not updated.get("lead_email"):
        email = extract_email(user_message)
        if email:
            updated["lead_email"] = email
        return updated

    if not updated.get("lead_platform"):
        platform = extract_platform(user_message)
        if platform:
            updated["lead_platform"] = platform
        elif user_message.strip():
            # Accept any non-empty answer as platform if no keyword matched
            updated["lead_platform"] = user_message.strip().capitalize()
        return updated

    return updated


# ─────────────────────────────────────────────
# 8. GRAPH NODES
# ─────────────────────────────────────────────

def intent_node(state: AgentState) -> AgentState:
    """Detect or update intent based on the latest user message."""
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    new_intent = detect_intent(last_human, state.get("intent", "casual"))

    updated = dict(state)
    updated["intent"] = new_intent

    # Start lead collection flow on high intent
    if new_intent == "high_intent" and not state.get("lead_captured"):
        updated["collecting_lead"] = True

    return updated


def extract_node(state: AgentState) -> AgentState:
    """Extract lead fields from the latest user message if we're collecting."""
    if not state.get("collecting_lead") or state.get("lead_captured"):
        return state

    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    return extract_lead_fields(state, last_human)


def tool_node(state: AgentState) -> AgentState:
    """Execute lead capture tool if all fields are collected."""
    if (state.get("collecting_lead")
            and state.get("lead_name")
            and state.get("lead_email")
            and state.get("lead_platform")
            and not state.get("lead_captured")):

        result = mock_lead_capture(
            state["lead_name"],
            state["lead_email"],
            state["lead_platform"],
        )
        updated = dict(state)
        updated["lead_captured"] = True
        updated["collecting_lead"] = False
        return updated

    return state


def response_node(state: AgentState) -> AgentState:
    """Generate the agent's response using the LLM with full context."""
    llm = get_llm()
    system_prompt = build_system_prompt(state)

    lc_messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = llm.invoke(lc_messages)

    updated = dict(state)
    updated["messages"] = state["messages"] + [AIMessage(content=response.content)]
    return updated


# ─────────────────────────────────────────────
# 9. ROUTING LOGIC
# ─────────────────────────────────────────────

def should_capture_lead(state: AgentState) -> Literal["tool", "respond"]:
    if (state.get("collecting_lead")
            and state.get("lead_name")
            and state.get("lead_email")
            and state.get("lead_platform")
            and not state.get("lead_captured")):
        return "tool"
    return "respond"


# ─────────────────────────────────────────────
# 10. BUILD THE GRAPH
# ─────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("extract", extract_node)
    graph.add_node("tool", tool_node)
    graph.add_node("respond", response_node)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "extract")
    graph.add_conditional_edges("extract", should_capture_lead, {
        "tool": "tool",
        "respond": "respond",
    })
    graph.add_edge("tool", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


# ─────────────────────────────────────────────
# 11. CONVERSATION RUNNER
# ─────────────────────────────────────────────

def run_agent():
    """Interactive CLI conversation loop."""
    app = build_graph()

    state: AgentState = {
        "messages": [],
        "intent": "casual",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_lead": False,
    }

    print("\n" + "="*60)
    print("  🎬 AutoStream AI Sales Assistant")
    print("  Type 'quit' or 'exit' to end the conversation.")
    print("="*60 + "\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("\nAssistant: Thanks for chatting! Have a great day. 🎬\n")
            break

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        state = app.invoke(state)

        last_ai = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            "..."
        )
        print(f"\nAssistant: {last_ai}\n")


if __name__ == "__main__":
    run_agent()
