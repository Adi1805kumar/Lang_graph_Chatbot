import os
import tempfile
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_PATH = "./chroma_store"

# Lightweight local embedding model — no API key needed
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(session_id: str):
    """Each chat session gets its own ChromaDB collection."""
    safe_id = session_id.replace("-", "_")
    return chroma_client.get_or_create_collection(
        name=f"sess_{safe_id}",
        embedding_function=emb_fn,
    )


def ingest_file(uploaded_file, session_id: str) -> int:
    """
    Accept a Streamlit UploadedFile, split into chunks,
    embed and store in ChromaDB. Returns number of chunks.
    """
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()

    # Write to a temp file so LangChain loaders can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=60
        )
        chunks = splitter.split_documents(docs)

        collection = get_collection(session_id)

        # Use filename+index as unique IDs so re-uploads don't crash
        ids    = [f"{filename}__{i}" for i in range(len(chunks))]
        texts  = [c.page_content for c in chunks]
        metas  = [{"source": filename, "chunk": i} for i in range(len(chunks))]

        collection.upsert(documents=texts, ids=ids, metadatas=metas)
        return len(chunks)
    finally:
        os.unlink(tmp_path)


def retrieve(query: str, session_id: str, k: int = 4) -> list[str]:
    """Return the top-k relevant text chunks for a query."""
    try:
        collection = get_collection(session_id)
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_texts=[query],
            n_results=min(k, count)
        )
        return results["documents"][0] if results["documents"] else []
    except Exception:
        return []