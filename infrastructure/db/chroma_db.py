import re
import uuid
from typing import List, Dict
from pathlib import Path

from chromadb import PersistentClient, QueryResult
from sentence_transformers import SentenceTransformer

from core.config import settings
from core.models.document import Document, DocumentChunk
from infrastructure.db.keyword_indexer import KeywordIndexer
from infrastructure.db.vector_db import IVectorDatabase


class ChromaDB(IVectorDatabase):
    def __init__(self):
        persist_path = str(settings.CHROMADB_DIR.absolute())
        Path(persist_path).mkdir(parents=True, exist_ok=True)
        print(f"ChromaDB init called {persist_path}")

        self.client = PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection('knowledgebase')
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.keyword_indexer = KeywordIndexer()

    def add_documents(self, documents: List[Document]) -> None:
        for document in documents:
            chunks = self._spit_into_paragraphs(document)
            embeddings = self.embedding_model.encode([chunk.content for chunk in chunks])
            chunk_ids = [f"doc_{i}_{uuid.uuid4()}" for i in range(len(chunks))]

            self.collection.add(
                documents=[chunk.content for chunk in chunks],
                embeddings=embeddings.tolist(),
                metadatas=[{
                    "role": document.role,
                    "source": document.source_url,
                    "section": chunk.section_title
                } for chunk in chunks],
                ids=chunk_ids,
            )

            for chunk, chunk_id in zip(chunks, chunk_ids):
                self.keyword_indexer.index_keywords(chunk, chunk_id)

    def search(self, query: str, filter_roles: List[str], top_k: int = 3) -> List[str]:
        keywords = self.keyword_indexer.search(query)
        candidate_ids = set()

        for kw in keywords:
            candidate_ids.update(self.keyword_indexer.keyword_map.get(kw, []))

        results = self.collection.query(
            query_embeddings=[self.embedding_model.encode(query).tolist()],
            n_results=top_k * 5,  # запас для фильтрации
            where={"role": {"$in": filter_roles}},
        )

        return self._post_filter_results(results, candidate_ids, top_k)

    def _spit_into_paragraphs(self, doc: Document) -> List[DocumentChunk]:
        raw_paragraphs = re.split(r'\n\n+', doc.content)
        chunks = []
        current_section = "General"

        for paragraph in raw_paragraphs:
            if re.match(r'^\*\*.+\*\*$', paragraph.strip()):
                current_section = paragraph.strip('*').strip()
                continue

            if not paragraph.strip():
                continue

            if len(paragraph) > settings.CHUNK_SIZE:
                sub_chunks = self._split_long_paragraph(paragraph)
                for sc in sub_chunks:
                    chunks.append(DocumentChunk(
                        content=sc,
                        section_title=current_section,
                        metadata={"source": doc.source_url}
                    ))
            else:
                chunks.append(DocumentChunk(
                    content=paragraph,
                    section_title=current_section,
                    metadata={"source": doc.source_url}
                ))

        return chunks

    def _split_long_paragraph(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= settings.CHUNK_SIZE:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _post_filter_results(self, results: QueryResult, candidate_ids: set, top_k: int) -> List[str]:
        documents = results['documents'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]

        filtered_results = []
        seen_docs = set()

        for doc, dist, doc_id in zip(documents, distances, ids):
            if not candidate_ids or doc_id in candidate_ids:
                if doc not in seen_docs:
                    seen_docs.add(doc)
                    filtered_results.append((doc, dist))

        filtered_results.sort(key=lambda x: x[1])
        return [doc for doc, _ in filtered_results[:top_k]]

    def clear_collection(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(self.collection.name)
        self.keyword_indexer.keyword_map.clear()
