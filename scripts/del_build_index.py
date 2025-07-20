import os
import json
import re
import uuid
import sqlite3
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# === Init embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Paths ===
DOCS_DIR = Path("docs")
DB_PATH = "memory.sqlite"

# === Chunking strategies ===
def detect_chunking_strategy(content: str) -> str:
    word_count = len(content.split())
    if re.search(r"(?m)^\s*\d+[\.\)]\s+", content) or len(re.findall(r"(?m)^[-•*]\s+", content)) >= 3:
        return "single"
    if word_count > 1000:
        return "sliding"
    return "split"

def chunk_single(content: str) -> List[str]:
    return [content]

def chunk_split_by_paragraph(content: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", content)
    return [p.strip() for p in paragraphs if len(p.strip()) > 50]

def chunk_sliding_window(content: str, window=100, step=60) -> List[str]:
    words = content.split()
    return [" ".join(words[i:i+window]) for i in range(0, len(words), step)]

def chunk_content(content: str) -> List[str]:
    strategy = detect_chunking_strategy(content)
    if strategy == "single":
        return chunk_single(content)
    elif strategy == "split":
        return chunk_split_by_paragraph(content)
    elif strategy == "sliding":
        return chunk_sliding_window(content)
    else:
        raise ValueError("Unknown strategy")

# === Save to SQLite ===
def save_chunks_to_db(chunks: List[Dict]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
              CREATE TABLE IF NOT EXISTS chunks (
                                                    id TEXT PRIMARY KEY,
                                                    text TEXT,
                                                    embedding BLOB,
                                                    metadata TEXT
              )
              """)

    for ch in chunks:
        c.execute("INSERT INTO chunks (id, text, embedding, metadata) VALUES (?, ?, ?, ?)", (
            ch["id"],
            ch["text"],
            ch["embedding"],
            json.dumps(ch["metadata"])
        ))

    conn.commit()
    conn.close()

# === Main pipeline ===
def process_documents():
    all_chunks = []

    for file in sorted(DOCS_DIR.glob("doc_*.json")):
        with open(file, "r", encoding="utf-8") as f:
            doc = json.load(f)

        content = doc.get("content", "")
        text_chunks = chunk_content(content)

        embeddings = model.encode(text_chunks)
        for i, text in enumerate(text_chunks):
            chunk_id = str(uuid.uuid4())
            chunk = {
                "id": chunk_id,
                "text": text,
                "embedding": embeddings[i].tobytes(),
                "metadata": {
                    "doc_id": file.stem,
                    "title": doc.get("title", ""),
                    "role": doc.get("role", "default"),
                    "source_url": doc.get("source_url", "")
                }
            }
            all_chunks.append(chunk)
        print(f"✅ {file.name}: {len(text_chunks)} chunks")

    save_chunks_to_db(all_chunks)
    print(f"\n📦 Total chunks saved: {len(all_chunks)} → {DB_PATH}")

# === Run ===
if __name__ == "__main__":
    process_documents()
