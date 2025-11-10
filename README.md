# MechaBot — Smart Documentation Assistant (RAG)

An **intelligent bot over your docs**: it ingests Markdown/README/notes from a folder, builds a vector index,
and answers questions **with source citations**. Built for quick demos and internship portfolios.

## ✨ Features
- Local vector store using **ChromaDB**
- Embeddings via **Sentence-Transformers** (`all-MiniLM-L6-v2`) — runs on CPU
- Streamlit chat UI with chat history
- Sources panel with chunk-level citations
- Pluggable LLM: **OpenAI** via `OPENAI_API_KEY` (optional). If not set, MechaBot returns retrieved chunks only.

## 🧱 Project layout
```
MechaBot/
├─ app/
│  └─ app.py                 # Streamlit chat UI
├─ data/sample_docs/         # Example docs (ingest these first)
│  ├─ README.md
│  └─ config.md
├─ scripts/
│  └─ ingest.py              # Build vector index from a folder
├─ eval/qna.json             # Tiny evaluation set (optional)
├─ requirements.txt
└─ vectorstore/              # Created by Chroma (after ingest)
```

## 🚀 Quickstart
```bash
# 1) Create & activate a virtual env (recommended)
python -m venv .venv && source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Set your LLM key
export OPENAI_API_KEY=sk-...                         # Windows PowerShell: $Env:OPENAI_API_KEY="sk-..."

# 4) Ingest docs (use the sample folder or your own)
python scripts/ingest.py --docs data/sample_docs --persist_dir vectorstore

# 5) Run the chat UI
streamlit run app/app.py
```

Open the local URL Streamlit prints (e.g., http://localhost:8501).

## 🧪 Minimal eval (optional)
```bash
python scripts/ingest.py --docs data/sample_docs --persist_dir vectorstore
# Open the app and ask the questions in eval/qna.json; verify answers & citations.
```

## 🛠️ Notes
- You can point `--docs` to any folder. Supported types: `.md`, `.txt`. (You can extend to PDFs.)
- If `OPENAI_API_KEY` is **not** set, the app will show top retrieved chunks as the "answer" (good for offline demos).
- For PDFs, consider `pymupdf` or `pypdf` + text extraction, then add to `ALLOWED_EXTS` in `ingest.py`.

## 🧭 Why this project matches “KI-gestützte Softwareentwicklung / Textanalyse / intelligente Bots”
- **KI-gestützte Softwareentwicklung**: RAG over engineering docs/README/Runbooks speeds onboarding & dev workflows.
- **Analyse von Textdaten**: ingestion, chunking, embeddings, similarity search, and qualitative eval.
- **Entwicklung intelligenter Bots**: a working assistant with citations and a UI.
