# Renovex Chat Assistant (RAG)

A **documentation-aware chatbot** for Renovex Hausmodernisierung GmbH.  
It ingests the company’s service information (Markdown files), builds a vector index, and answers customer questions **based on that documentation**.  
Supports OpenAI or local models (Ollama).

## ✨ Features
- Retrieval-Augmented Generation (**RAG**) over your docs
- **ChromaDB** vector store (stored locally)
- Embeddings via **Sentence-Transformers** (`all-MiniLM-L6-v2`)
- Streamlit web UI with chat history
- Works **with or without** external LLM API keys
- When no LLM is configured → shows relevant source text directly

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
# 1) Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Enable OpenAI LLM responses
export OPENAI_API_KEY=sk-...

# 4) Ingest Renovex documentation
python scripts/ingest.py --docs data/project_docs --persist_dir vectorstore --collection renovex

# 5) Start the chat interface
streamlit run app/app.py

Open the local URL Streamlit prints (e.g., http://localhost:8501).

To use local models instead of OpenAI:
ollama pull llama3

Remove the vectorstore to start fresh
rm -rf vectorstore

## 🧪 Minimal eval (optional)
```bash
python scripts/ingest.py --docs data/sample_docs --persist_dir vectorstore
# Open the app and ask the questions in eval/qna.json; verify answers & citations.
```
