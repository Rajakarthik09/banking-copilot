"""Streamlit chat UI. Talks to the API only (no direct DB / LLM access)."""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
# Demo RM -> API key mapping, same format as the API's API_KEYS ("key:RM-1,...").
KEYS = {rm.strip(): key.strip() for key, rm in
        (p.split(":", 1) for p in os.environ.get(
            "API_KEYS", "rm1-demo-key:RM-1,rm2-demo-key:RM-2,rm3-demo-key:RM-3,rm4-demo-key:RM-4,"
                        "rm5-demo-key:RM-5").split(",") if ":" in p)}
MODES = {
    "tools_rag": "Tools + policies (function calling + RAG)",
    "tools": "Tools only (function calling)",
    "rag": "Policies only (RAG)",
    "prompt_only": "LLM only (no data, baseline)",
}
EXAMPLES = [
    "Which SME loans in my portfolio are more than 30 days past due, and what's the exposure?",
    "How many fee waivers can a customer get per year?",
    "Why was transaction {txn} flagged and which policy applies?",
    "Show me customer C0002.",
]

st.set_page_config(page_title="Banking Copilot", page_icon="🏦", layout="wide")

with st.sidebar:
    st.header("Banking Copilot")
    rm = st.selectbox("Signed in as", sorted(KEYS), help="Demo login: each RM has its own API key")
    mode = st.radio("Mode", list(MODES), format_func=MODES.get)
    verify = st.toggle("Grounding verifier", value=True,
                       help="Remove sentences whose numbers or citations aren't backed by sources")
    try:
        me = httpx.get(f"{API_URL}/me", headers={"X-API-Key": KEYS[rm]}, timeout=5).json()
        st.caption(f"{me.get('name', rm)} · {me.get('region', '')}")
        health = httpx.get(f"{API_URL}/health", timeout=5).json()
        st.caption(f"API: {health.get('status')} · LLM: {health.get('llm')} · chunks: {health.get('chunks')}")
    except httpx.HTTPError:
        st.error(f"API not reachable at {API_URL}")
    if st.button("Clear chat"):
        st.session_state.pop(f"history_{rm}", None)

history = st.session_state.setdefault(f"history_{rm}", [])  # per RM, so chats never mix


def render_details(r: dict) -> None:
    status = "✅ verified" if r["verified"] else "⚠️ not fully verified"
    if r["refused"]:
        status += f" · refused ({r['refusal_reason']})"
    with st.expander(f"Sources · tools · {status} · {r['latency_ms'] / 1000:.1f}s · {r['tokens']} tokens"):
        if r["tools_called"]:
            st.markdown("**Tools called**")
            for t in r["tools_called"]:
                st.code(f"{t['name']}({t['args']})  ->  {'ok' if t['ok'] else t['error']}", language="text")
        if r["sources"]:
            st.markdown("**Policy sources**: " + ", ".join(sorted({s["source"] for s in r["sources"]})))
        if r["removed"]:
            st.markdown("**Removed by the verifier**")
            for s in r["removed"]:
                st.markdown(f"- ~~{s}~~")


for turn in history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if "details" in turn:
            render_details(turn["details"])

st.caption("Try: " + " · ".join(f"_{e}_" for e in EXAMPLES[:2]))
if question := st.chat_input("Ask about policies or your portfolio"):
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            resp = httpx.post(f"{API_URL}/chat", headers={"X-API-Key": KEYS[rm]},
                              json={"question": question, "mode": mode, "verify": verify}, timeout=600)
            resp.raise_for_status()
            r = resp.json()
            st.markdown(r["answer"])
            render_details(r)
            history.append({"role": "assistant", "content": r["answer"], "details": r})
        except httpx.HTTPStatusError as e:
            st.error(f"API error {e.response.status_code}: {e.response.json().get('detail', e.response.text)}")
        except httpx.HTTPError as e:
            st.error(f"Could not reach the API: {e}")
