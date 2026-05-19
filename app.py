import streamlit as st
from dotenv import load_dotenv

from database import (
    init_db, create_session, get_all_sessions,
    delete_session, get_messages, save_message,
    get_documents, save_document,
)
from rag import ingest_file
from graph import stream_graph

load_dotenv()
init_db()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main background */
.stApp { background-color: #0f1117; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1a1d27;
    border-right: 1px solid #2e3149;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: #1e2235;
    border-radius: 12px;
    border: 1px solid #2e3149;
    margin-bottom: 8px;
    padding: 4px 8px;
}

/* User message different shade */
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #2a2060;
}

/* Input box */
[data-testid="stChatInput"] textarea {
    background-color: #1e2235 !important;
    border: 1px solid #2e3149 !important;
    border-radius: 12px !important;
    color: #e8e9f0 !important;
}

/* Buttons */
.stButton > button {
    background-color: #7c6af7;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #6355d4;
    color: white;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #1e2235;
    border: 1px solid #2e3149;
    border-radius: 10px;
}

/* Info/success boxes */
.stAlert { border-radius: 10px; }

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #1e2235;
    border-radius: 10px;
    border: 1px dashed #2e3149;
    padding: 8px;
}

/* Sidebar session buttons */
div[data-testid="stSidebarContent"] .stButton > button {
    width: 100%;
    text-align: left;
    background-color: #252836;
    color: #e8e9f0;
    font-weight: 400;
    margin-bottom: 3px;
    border: 1px solid #2e3149;
}
div[data-testid="stSidebarContent"] .stButton > button:hover {
    background-color: #2e3350;
    border-color: #7c6af7;
}

/* Divider */
hr { border-color: #2e3149; }

/* Typography */
h1, h2, h3 { color: #e8e9f0 !important; }
p, li, label { color: #c8cad8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
# st.session_state persists across reruns within the same browser tab.
if "active_session_id" not in st.session_state:
    st.session_state.active_session_id = None

if "messages" not in st.session_state:
    # List of {"role": "user"/"assistant", "content": "..."}
    st.session_state.messages = []

if "use_rag" not in st.session_state:
    st.session_state.use_rag = False

if "uploading" not in st.session_state:
    st.session_state.uploading = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 LangGraph Chatbot")
    st.markdown("---")

    # New chat button
    if st.button("＋  New Chat", use_container_width=True):
        new_sess = create_session("New Chat")
        st.session_state.active_session_id = new_sess.id
        st.session_state.messages = []
        st.rerun()

    st.markdown("### 💬 Your Chats")

    # List all sessions
    sessions = get_all_sessions()
    if not sessions:
        st.caption("No chats yet. Click '+ New Chat' above.")
    else:
        for sess in sessions:
            col1, col2 = st.columns([5, 1])
            label = f"{sess['title']} ({sess['message_count']})"
            is_active = sess["id"] == st.session_state.active_session_id

            with col1:
                btn_style = "**" if is_active else ""
                if st.button(
                    f"{'▶ ' if is_active else ''}{sess['title']}",
                    key=f"sess_{sess['id']}",
                    use_container_width=True,
                    help=f"{sess['message_count']} messages",
                ):
                    st.session_state.active_session_id = sess["id"]
                    st.session_state.messages = get_messages(sess["id"])
                    st.rerun()

            with col2:
                if st.button("🗑", key=f"del_{sess['id']}", help="Delete"):
                    delete_session(sess["id"])
                    if st.session_state.active_session_id == sess["id"]:
                        st.session_state.active_session_id = None
                        st.session_state.messages = []
                    st.rerun()

    st.markdown("---")

    # ── RAG section (only shown when a session is active) ─────────────────────
    if st.session_state.active_session_id:
        st.markdown("### 📎 RAG Documents")

        # Toggle RAG on/off
        st.session_state.use_rag = st.toggle(
            "Enable RAG",
            value=st.session_state.use_rag,
            help="When ON, the bot answers using your uploaded documents.",
        )

        if st.session_state.use_rag:
            st.success("RAG is **ON** — answers from your docs", icon="🟢")
        else:
            st.info("RAG is **OFF** — regular chat mode", icon="⚪")

        # File uploader
        uploaded = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt", "md"],
            help="Upload a document to chat with it.",
            label_visibility="collapsed",
        )

        if uploaded is not None:
            with st.spinner(f"Processing **{uploaded.name}**…"):
                try:
                    count = ingest_file(uploaded, st.session_state.active_session_id)
                    save_document(st.session_state.active_session_id, uploaded.name)
                    st.success(f"✅ {uploaded.name} — {count} chunks indexed!")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

        # Show already-uploaded documents for this session
        docs = get_documents(st.session_state.active_session_id)
        if docs:
            st.markdown("**Indexed documents:**")
            for d in docs:
                st.markdown(f"📄 `{d['filename']}`")

    st.markdown("---")
    st.caption("Built with LangGraph · Groq · ChromaDB · SQLite")


# ── Main chat area ────────────────────────────────────────────────────────────
if not st.session_state.active_session_id:
    # Welcome screen when no session is selected
    st.markdown("<br>" * 3, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding: 40px;'>
            <h1>🤖 LangGraph Chatbot</h1>
            <p style='font-size:16px; color:#9499b5; margin-top:12px;'>
                A fully local AI chatbot with memory, streaming,<br>
                RAG document search, and session history.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ✨ Features")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            - 💬 **Chat with memory** — full history per session
            - 📂 **Resume any chat** — click a session to continue
            - ⚡ **Streaming** — see tokens as they arrive
            """)
        with col_b:
            st.markdown("""
            - 📎 **RAG** — upload PDFs and chat with them
            - 🗄️ **SQLite DB** — all chats saved locally
            - 🧠 **LangGraph** — structured agent pipeline
            """)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀  Start a New Chat", use_container_width=True):
            new_sess = create_session("New Chat")
            st.session_state.active_session_id = new_sess.id
            st.session_state.messages = []
            st.rerun()

else:
    # ── Active session: show chat ─────────────────────────────────────────────
    sessions = get_all_sessions()
    current = next(
        (s for s in sessions if s["id"] == st.session_state.active_session_id),
        None,
    )

    # Header
    title = current["title"] if current else "Chat"
    rag_badge = " 🟢 RAG" if st.session_state.use_rag else ""
    st.markdown(f"### {title}{rag_badge}")
    st.markdown("---")

    # ── Render existing messages ──────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask anything… (RAG is ON — using your docs)"
        if st.session_state.use_rag
        else "Ask anything…"
    )

    if user_input:
        # 1. Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Add to in-memory history and save to DB
        st.session_state.messages.append({"role": "user", "content": user_input})
        save_message(st.session_state.active_session_id, "user", user_input)

        # 3. Stream assistant response
        with st.chat_message("assistant"):
            # st.write_stream accepts a generator and streams tokens live
            response_tokens = stream_graph(
                history=st.session_state.messages,
                session_id=st.session_state.active_session_id,
                use_rag=st.session_state.use_rag,
            )
            full_response = st.write_stream(response_tokens)

        # 4. Save complete assistant reply to DB
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        save_message(
            st.session_state.active_session_id, "assistant", full_response
        )

        # 5. Rerun to refresh sidebar (updated title + message count)
        st.rerun()