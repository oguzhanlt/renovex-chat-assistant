import argparse, os, re, glob
from typing import List
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ALLOWED_EXTS = {'.md', '.txt'} # PDF will be added soon

def read_docs(root: str) -> List[tuple[str, str]]:
    docs = []
    for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if os.path.isdir(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext in ALLOWED_EXTS:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            docs.append((path, text))
    return docs

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> List[str]:
    # word-based chunker
    words = re.split(r"\s+", text.strip())
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
        if chunk_size <= overlap:
            break
    return [c for c in chunks if c.strip()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True, help="Folder with .md/.txt documents")
    parser.add_argument("--persist_dir", default="vectorstore", help="Chroma persistence dir")
    parser.add_argument("--collection", default="renovex", help="Chroma collection name")
    args = parser.parse_args()

    # loading encoder
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # initializing Chroma
    client = chromadb.PersistentClient(path=args.persist_dir, settings=Settings(allow_reset=True))
    collection = client.get_or_create_collection(args.collection)

    # clear collection
    try:
        collection.delete(where={})
    except Exception:
        pass

    # read
    items = read_docs(args.docs)
    ids, docs, metas = [], [], []
    idx = 0
    for path, text in items:
        # chunk each document
        for chunk in chunk_text(text):
            ids.append(f"doc_{idx}")
            docs.append(chunk)
            metas.append({"source": path})
            idx += 1

    # embed and store
    embeddings = model.encode(docs, normalize_embeddings=True).tolist()
    collection.add(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)

    print(f"Ingested {len(docs)} chunks from {len(items)} files into collection '{args.collection}'.")

if __name__ == "__main__":
    main()
