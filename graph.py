import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, BaseMessage
)
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from rag import retrieve

load_dotenv()

# ── LLM setup ────────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY", ""),
    streaming=True,
    temperature=0.7,
)

SYSTEM_PROMPT = """You are a helpful, friendly, and knowledgeable AI assistant.
You remember the full conversation history.
When document context is provided, use it to give accurate, grounded answers.
Be concise but thorough. Format your replies with markdown when helpful."""


# ── LangGraph State ───────────────────────────────────────────────────────────
# This TypedDict is the shared state that flows through all graph nodes.
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # full chat history
    session_id: str
    use_rag: bool
    rag_context: str       # filled by the retrieve node


# ── Node 1: RAG retrieval ─────────────────────────────────────────────────────
def retrieve_node(state: GraphState) -> dict:
    """
    If RAG is enabled, find the last human message and
    fetch relevant chunks from ChromaDB.
    """
    if not state.get("use_rag"):
        return {"rag_context": ""}

    last_human = next(
        (m.content for m in reversed(state["messages"])
         if isinstance(m, HumanMessage)),
        "",
    )
    chunks = retrieve(last_human, state["session_id"], k=4)
    context = "\n\n---\n\n".join(chunks) if chunks else ""
    return {"rag_context": context}


# ── Node 2: LLM chat ───────────────────────────────────────────────────────────
def chat_node(state: GraphState) -> dict:
    """
    Build the system prompt (with RAG context if available),
    call Groq, and return the AI reply as a new message.
    """
    system_content = SYSTEM_PROMPT
    if state.get("rag_context"):
        system_content += (
            "\n\n=== Relevant content from uploaded documents ===\n"
            + state["rag_context"]
            + "\n=================================================\n"
            "Use the above context when answering."
        )

    full_messages = [SystemMessage(content=system_content)] + state["messages"]
    response = llm.invoke(full_messages)
    return {"messages": [response]}


# ── Conditional router ────────────────────────────────────────────────────────
def route(state: GraphState) -> str:
    """Decide whether to run the RAG node first or go straight to chat."""
    return "retrieve" if state.get("use_rag") else "chat"


# ── Build & compile the graph ─────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("chat", chat_node)

    # Conditional entry: RAG path or direct chat path
    builder.set_conditional_entry_point(
        route,
        {"retrieve": "retrieve", "chat": "chat"},
    )
    builder.add_edge("retrieve", "chat")
    builder.add_edge("chat", END)

    return builder.compile()


graph = build_graph()


# ── Public helpers called by app.py ──────────────────────────────────────────

def run_graph(history: list[dict], session_id: str, use_rag: bool) -> str:
    """
    Run the full graph (non-streaming).
    history: list of {"role": "user"/"assistant", "content": "..."}
    """
    lc_messages = _to_lc_messages(history)
    state = {
        "messages": lc_messages,
        "session_id": session_id,
        "use_rag": use_rag,
        "rag_context": "",
    }
    result = graph.invoke(state)
    return result["messages"][-1].content


def stream_graph(history: list[dict], session_id: str, use_rag: bool):
    """
    Generator that yields string tokens one by one.
    Handles RAG retrieval first, then streams LLM output.
    """
    from rag import retrieve as rag_retrieve

    rag_context = ""
    if use_rag and history:
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"),
            "",
        )
        chunks = rag_retrieve(last_user, session_id, k=4)
        rag_context = "\n\n---\n\n".join(chunks) if chunks else ""

    system_content = SYSTEM_PROMPT
    if rag_context:
        system_content += (
            "\n\n=== Relevant content from uploaded documents ===\n"
            + rag_context
            + "\n=================================================\n"
            "Use the above context when answering."
        )

    lc_messages = [SystemMessage(content=system_content)] + _to_lc_messages(history)

    for chunk in llm.stream(lc_messages):
        if chunk.content:
            yield chunk.content


def _to_lc_messages(history: list[dict]) -> list[BaseMessage]:
    """Convert simple dicts to LangChain message objects."""
    result = []
    for m in history:
        if m["role"] == "user":
            result.append(HumanMessage(content=m["content"]))
        else:
            result.append(AIMessage(content=m["content"]))
    return result