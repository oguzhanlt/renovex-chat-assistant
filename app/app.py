import os
import streamlit as st
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from openai import OpenAI

st.set_page_config(page_title="Renovex • Chat Assistant", layout="wide")

# config
PERSIST_DIR = os.environ.get("RENOVEX_PERSIST_DIR", "vectorstore")
COLLECTION = os.environ.get("RENOVEX_COLLECTION", "renovex")
TOP_K = int(os.environ.get("RENOVEX_TOP_K", "4"))
MODEL_NAME = os.environ.get("RENOVEX_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # can be plugged in, but default is to use local LLM

# initializing vector store and encoders
@st.cache_resource
def get_vs():
    client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(allow_reset=False))
    collection = client.get_or_create_collection(COLLECTION)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return collection, embedder

collection, embedder = get_vs()

def retrieve(query: str, k: int = TOP_K):
    q = embedder.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q, n_results=k, include=["documents", "metadatas", "distances", "embeddings"])
    return res

def build_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    return f"""You are a helpful assistant answering from the provided documentation.
Cite sources using [#] and keep answers concise.

Context:
{context_block}

Question: {question}

Answer with references like [1], [2] corresponding to the chunks."""

import subprocess

def call_llm(prompt: str, contexts: list[str], metas: list[dict]) -> str:
    joined_text = "\n".join(contexts)
    if not joined_text.strip():
        return "No relevant information found in the loaded documentation."

    full_prompt = f"You are an assistant answering questions from documentation.\n\nContext:\n{joined_text}\n\nQuestion:\n{prompt}\n\nAnswer concisely with references if possible."

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error using local LLM: {e}"

# --- UI ---
st.title("🤖 Renovex — Smart Documentation Assistant")
st.caption("RAG over your docs with citations. Ingest your folder via `scripts/ingest.py`.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {q, answer, sources}

with st.sidebar:
    st.header("Settings")
    st.write(f"Vector store: `{PERSIST_DIR}` / Collection: `{COLLECTION}`")
    top_k = st.slider("Top-K", min_value=1, max_value=10, value=TOP_K)
    st.write("LLM:", "OpenAI (via `OPENAI_API_KEY`)" if OPENAI_API_KEY else "Ollama (offline mode)")
    st.divider()
    st.markdown("**How to use**")
    st.markdown("1) Run ingest script.\n2) Ask a question.\n3) See citations in the right panel.")

col_left, col_right = st.columns([2, 1])

with col_left:
    question = st.text_input("Ask about your docs:", placeholder="How do I disable moisture reduction for a dry run?")
    if st.button("Ask", type="primary") and question.strip():
        res = retrieve(question, k=top_k)
        contexts = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        prompt = build_prompt(question, contexts)
        answer = call_llm(prompt, contexts, metadatas)
        st.session_state.history.append({"q": question, "answer": answer, "contexts": contexts, "metas": metadatas})

    for turn in st.session_state.history[::-1]:
        st.markdown(f"**You:** {turn['q']}")
        st.markdown(f"**Renovex:** {turn['answer']}")
        st.divider()

with col_right:
    st.subheader("Sources")
    if st.session_state.history:
        latest = st.session_state.history[-1]
        for i, (ctx, meta) in enumerate(zip(latest["contexts"], latest["metas"])):
            st.markdown(f"**[{i+1}]** *{meta.get('source','unknown')}*")
            with st.expander("Show chunk"):
                st.write(ctx)
