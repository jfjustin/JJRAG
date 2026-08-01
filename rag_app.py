"""
JJRAG — Document Q&A with a switchable model backend.

Two modes, chosen from a dropdown in the UI:

  🔒 Private (Local)   — open-source models via Ollama + local embeddings.
                         Nothing leaves the machine. No API key required.
  🎯 Accuracy (Cloud)  — the current best hosted model (Claude Opus 4.8 by
                         default, OpenAI optional). You paste your OWN API key;
                         it is held in session memory only, never written to
                         disk and never logged.

Built on top of the original JJrag_final.py retrieval prototype, extended with
the missing generation step so the app actually answers questions instead of
just dumping raw chunks.
"""

import os

# Force the torch-only backend for transformers/sentence-transformers. Without
# this, importing the local embedding model can crash on machines with Keras 3
# installed (transformers tries to load its TensorFlow backend). Must be set
# before any transformers import happens.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import logging
import traceback
from pathlib import Path

import requests
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# Logging — note: we deliberately never log API keys or document contents.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("rag_app.log"), logging.StreamHandler()],
)
logger = logging.getLogger("jjrag")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEMP_DIR = "temp_docs"
TOP_K = 4


# ===========================================================================
# Step 1 — document ingestion  (load -> split -> embed -> vector store)
# ===========================================================================
def load_documents(uploaded_files):
    """Read uploaded PDF/DOCX/TXT files into LangChain Document objects."""
    documents = []
    if not uploaded_files:
        st.warning("Please upload at least one document file.")
        return []

    os.makedirs(TEMP_DIR, exist_ok=True)
    for uploaded_file in uploaded_files:
        try:
            file_name = uploaded_file.name
            ext = Path(file_name).suffix.lower()
            temp_path = os.path.join(TEMP_DIR, file_name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if ext == ".pdf":
                loader = PyPDFLoader(temp_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(temp_path)
            elif ext == ".txt":
                loader = TextLoader(temp_path)
            else:
                st.warning(f"Skipping unsupported file: {file_name}")
                continue

            logger.info("Loading %s", ext)
            documents.extend(loader.load())
        except Exception as e:  # noqa: BLE001 - surface per-file errors, keep going
            logger.error("Error processing %s: %s", uploaded_file.name, e)
            logger.error(traceback.format_exc())
            st.error(f"Error loading {uploaded_file.name}: {e}")

    if documents:
        st.success(f"Loaded {len(documents)} document section(s).")
    else:
        st.warning("No documents were loaded.")
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    """Chunk documents into overlapping windows suitable for retrieval.

    The original prototype used chunk_size=40, which is far too small for
    meaningful retrieval — a 40-character window rarely holds a full sentence.
    1000/150 is a sensible default for prose documents.
    """
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Created %d chunks", len(chunks))
    return chunks


@st.cache_resource(show_spinner=False)
def _get_local_embeddings():
    """Local sentence-transformer embeddings — run once, cached across reruns.

    Used for BOTH modes so document text never has to leave the machine just to
    be indexed. Only the (optional) generation step in Accuracy mode calls out.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=LOCAL_EMBED_MODEL)


def create_vector_store(document_chunks):
    """Embed chunks locally and build an in-memory FAISS index."""
    if not document_chunks:
        st.warning("No document chunks available for indexing.")
        return None
    try:
        embeddings = _get_local_embeddings()
        return FAISS.from_documents(document_chunks, embeddings)
    except Exception as e:  # noqa: BLE001
        logger.error("Vector store error: %s", e)
        logger.error(traceback.format_exc())
        st.error(f"Error creating vector store: {e}")
        return None


# ===========================================================================
# Step 2 — generation backends
# ===========================================================================
SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions using ONLY the provided "
    "document excerpts. If the answer is not contained in the excerpts, say so "
    "plainly instead of guessing. Cite the relevant excerpt numbers in your "
    "answer where helpful."
)


def _build_prompt(question, docs):
    context = "\n\n".join(
        f"[Excerpt {i + 1}]\n{d.page_content}" for i, d in enumerate(docs)
    )
    return (
        f"Here are the document excerpts:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the excerpts above."
    )


def list_ollama_models():
    """Return locally-available Ollama model names (empty list if unreachable)."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:  # noqa: BLE001 - Ollama simply not running
        return []


def stream_local_answer(question, docs, model):
    """Stream an answer from a local Ollama model. 100% on-device."""
    prompt = _build_prompt(question, docs)
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": True,
    }
    with requests.post(
        f"{OLLAMA_HOST}/api/generate", json=payload, stream=True, timeout=600
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            import json

            chunk = json.loads(line)
            if chunk.get("response"):
                yield chunk["response"]


def stream_anthropic_answer(question, docs, api_key, model="claude-opus-4-8"):
    """Stream an answer from Anthropic's Claude. Key stays in memory only."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    prompt = _build_prompt(question, docs)
    with client.messages.stream(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def stream_openai_answer(question, docs, api_key, model="gpt-4o", base_url=None):
    """Stream an answer from any OpenAI-compatible endpoint. Key stays in memory only.

    ``base_url`` lets you point at a self-hosted, OpenAI-compatible server instead
    of api.openai.com — for example a vLLM instance serving **Kimi** on an AMD
    Instinct GPU in AMD Developer Cloud (``http://<droplet-ip>:8000/v1``). When it
    is ``None`` the client talks to OpenAI as usual. See the README section
    "Running Kimi on AMD Developer Cloud" for the full setup.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url or None)
    prompt = _build_prompt(question, docs)
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ===========================================================================
# Streamlit app
# ===========================================================================
def main():
    st.set_page_config(page_title="JJRAG — Document Q&A", page_icon="📄")
    st.title("📄 JJRAG — Document Q&A")
    st.caption(
        "Upload documents, pick a model backend, and ask questions grounded in "
        "your files."
    )

    # ---- Sidebar: mode + key + model selection ----------------------------
    with st.sidebar:
        st.header("⚙️ Model backend")
        mode = st.selectbox(
            "Choose how answers are generated",
            (
                "🔒 Private (Local — open-source, no data leaves your machine)",
                "🎯 Accuracy (Cloud API — best hosted model)",
            ),
        )
        is_private = mode.startswith("🔒")

        provider = None
        model_name = None
        api_key = None
        base_url = None

        if is_private:
            st.markdown(
                "Runs entirely on your machine via **Ollama**. "
                "No API key, no network calls, full privacy."
            )
            local_models = list_ollama_models()
            if local_models:
                model_name = st.selectbox("Local model", local_models)
            else:
                st.error(
                    "No Ollama models found. Install Ollama and run e.g. "
                    "`ollama pull llama3` (or `ollama pull deepseek-r1:8b`)."
                )
        else:
            provider = st.selectbox(
                "Provider",
                (
                    "Anthropic (Claude)",
                    "OpenAI (GPT)",
                    "Kimi on AMD Developer Cloud (OpenAI-compatible)",
                ),
            )
            # The private-key slot. type="password" masks it; it is stored ONLY
            # in st.session_state for this browser session and is never written
            # to disk or logged.
            api_key = st.text_input(
                "🔑 Your private API key",
                type="password",
                help=(
                    "Held in session memory only — never saved to disk, never "
                    "logged, never sent anywhere except the provider you chose. "
                    "For a self-hosted vLLM endpoint, use whatever token you "
                    "launched the server with (or any placeholder if it has none)."
                ),
                value=st.session_state.get("api_key", ""),
            )
            if api_key:
                st.session_state["api_key"] = api_key
            if provider.startswith("Anthropic"):
                model_name = st.selectbox(
                    "Model",
                    ("claude-opus-4-8", "claude-sonnet-5", "claude-haiku-4-5"),
                )
            elif provider.startswith("Kimi"):
                # Self-hosted, OpenAI-compatible endpoint (e.g. vLLM serving Kimi
                # on an AMD Instinct GPU in AMD Developer Cloud). Point base_url
                # at that server; the OpenAI backend handles the rest.
                base_url = st.text_input(
                    "Endpoint base URL",
                    value=os.getenv("OPENAI_BASE_URL", ""),
                    placeholder="http://<your-amd-droplet-ip>:8000/v1",
                    help=(
                        "Your vLLM OpenAI-compatible URL — see the README section "
                        "'Running Kimi on AMD Developer Cloud'."
                    ),
                )
                model_name = st.text_input(
                    "Model", value="moonshotai/Kimi-K2-Instruct"
                )
            else:
                base_url = os.getenv("OPENAI_BASE_URL") or None
                model_name = st.text_input("Model", value="gpt-4o")

        st.divider()
        st.caption(
            "Embeddings always run locally — your document text is never sent "
            "to a third party just to be indexed."
        )

    # ---- Document upload + processing -------------------------------------
    uploaded_files = st.file_uploader(
        "Add documents to process",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if st.button("📥 Process documents"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Loading, chunking, and indexing…"):
                docs = load_documents(uploaded_files)
                chunks = split_documents(docs)
                vs = create_vector_store(chunks)
                if vs is not None:
                    st.session_state.vector_store = vs
                    st.success(f"Indexed {len(chunks)} chunks. Ask away below.")

    # ---- Question + answer ------------------------------------------------
    query = st.text_input("Ask a question about your documents")

    if st.button("💬 Ask") and query:
        is_self_hosted = provider is not None and provider.startswith("Kimi")
        if st.session_state.vector_store is None:
            st.warning("Process some documents first.")
            return
        if not is_private and not api_key and not is_self_hosted:
            st.warning("Enter your API key in the sidebar to use Accuracy mode.")
            return
        if is_self_hosted and not base_url:
            st.warning("Enter your endpoint base URL in the sidebar.")
            return
        if is_private and not model_name:
            st.warning("No local model available — see the sidebar.")
            return

        docs = st.session_state.vector_store.similarity_search(query, k=TOP_K)

        st.subheader("Answer")
        try:
            if is_private:
                gen = stream_local_answer(query, docs, model_name)
            elif provider.startswith("Anthropic"):
                gen = stream_anthropic_answer(query, docs, api_key, model_name)
            else:
                # OpenAI (GPT) and Kimi-on-AMD share the OpenAI-compatible path.
                # Self-hosted vLLM often needs no real key — send a placeholder.
                gen = stream_openai_answer(
                    query, docs, api_key or "EMPTY", model_name, base_url=base_url
                )
            st.write_stream(gen)
        except Exception as e:  # noqa: BLE001
            # Never echo the API key, even on error.
            logger.error("Generation error: %s", type(e).__name__)
            st.error(f"Generation failed: {e}")

        with st.expander("📚 Sources used"):
            for i, d in enumerate(docs):
                src = d.metadata.get("source", "document")
                st.markdown(f"**Excerpt {i + 1}** — `{os.path.basename(src)}`")
                st.write(d.page_content)
                st.divider()


if __name__ == "__main__":
    main()
