import logging
import re
import uuid
from typing import List, Dict
from pathlib import Path

from chromadb import PersistentClient, QueryResult
from sentence_transformers import SentenceTransformer

from application.config import settings
from core.models.document import Document, DocumentChunk
from infrastructure.db.keyword_indexer import KeywordIndexer
from infrastructure.db.vector_db import IVectorDatabase

logger = logging.getLogger(__name__)


class ChromaDB(IVectorDatabase):
    def __init__(self):
        persist_path = str(settings.CHROMADB_DIR.absolute())
        Path(persist_path).mkdir(parents=True, exist_ok=True)

        self.client = PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection('knowledgebase')
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)

        self.keyword_indexer = KeywordIndexer(cache_path=settings.KEYWORDS_FILE)

        logger.info("ChromaDB initialized. Collection loaded with embedding model '%s'", settings.EMBEDDING_MODEL)

    def add_documents(self, documents: List[Document]) -> None:
        logger.info("Adding %d documents to ChromaDB...", len(documents))
        for document in documents:
            chunks = self._spit_into_paragraphs(document)

            title_chunk = DocumentChunk(
                content=document.title.strip(),
                section_title="Document Title",
                metadata={"source": document.source_url}
            )
            chunks.insert(0, title_chunk)

            texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_model.encode(texts)
            chunk_ids = [f"doc_{i}_{uuid.uuid4()}" for i in range(len(chunks))]

            self.collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=[{
                    "role": document.role,
                    "source": document.source_url,
                    "section": chunk.section_title,
                    "order": i
                } for i, chunk in enumerate(chunks)],
                ids=chunk_ids,
            )

            for chunk, chunk_id in zip(chunks, chunk_ids):
                logger.debug("Indexing chunk %s (section: %s)", chunk_id, chunk.section_title)
                self.keyword_indexer.index_keywords(chunk, chunk_id)

        self.keyword_indexer.save_cache()
        logger.info("Keyword indexing completed and cache saved.")

    def search(self, query: str, filter_roles: List[str], top_k: int = 3) -> List[Dict]:
        logger.info("Searching for query: '%s'", query)
        keywords = self.keyword_indexer.search(query)
        candidate_ids = set()

        logger.debug("Extracted keywords: %s", keywords)
        logger.debug("Keyword map keys: %s", list(self.keyword_indexer.keyword_map.keys())[:20])

        for kw in keywords:
            if kw in self.keyword_indexer.keyword_map:
                logger.debug("Keyword found in index: '%s'", kw)
            else:
                logger.debug("Keyword not found in index: '%s'", kw)

        for kw in keywords:
            candidate_ids.update(self.keyword_indexer.keyword_map.get(kw, []))

        strict_mode = False
        if not candidate_ids:
            logger.info("No keyword match found. Fallback to strict vector filtering.")
            strict_mode = True

        results = self.collection.query(
            query_embeddings=[self.embedding_model.encode(query).tolist()],
            n_results=top_k * 5,
            where={"role": {"$in": filter_roles}},
        )

        return self._post_filter_results(
            results,
            candidate_ids if candidate_ids else None,
            top_k,
            keywords,
            strict_mode
        )

    def _post_filter_results(
            self,
            results: QueryResult,
            candidate_ids: set | None,
            top_k: int,
            keywords: List[str],
            strict_mode: bool = False
    ) -> List[Dict]:
        documents = results['documents'][0]
        distances = results['distances'][0]
        ids = results['ids'][0]
        metadatas = results['metadatas'][0]

        filtered_results = []
        seen_docs = set()
        DISTANCE_THRESHOLD = 0.8

        for doc, dist, doc_id, meta in zip(documents, distances, ids, metadatas):
            if dist > DISTANCE_THRESHOLD:
                continue
            if candidate_ids and doc_id not in candidate_ids:
                continue

            if strict_mode and keywords:
                doc_lower = doc.lower()
                if not any(kw in doc_lower for kw in keywords):
                    logger.debug("Skipping doc_id=%s due to missing keywords in strict mode.", doc_id)
                    continue

            if doc not in seen_docs:
                seen_docs.add(doc)
                filtered_results.append({
                    "content": doc,
                    "distance": dist,
                    "source_url": meta.get("source", "unknown"),
                    "section": meta.get("section", "unknown"),
                    "role": meta.get("role", "unknown")
                })

        logger.info("Returning %d filtered result(s).", len(filtered_results[:top_k]))
        return filtered_results[:top_k]

    def get_chunks_by_source(self, source_url: str) -> List[str]:
        result = self.collection.get(where={"source": source_url})
        chunks = list(zip(result["documents"], result["metadatas"]))
        chunks.sort(key=lambda x: x[1].get("order", 0))
        return [doc for doc, _ in chunks]

    def clear_collection(self) -> None:
        logger.warning("Clearing vector collection and keyword index...")
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(self.collection.name)
        self.keyword_indexer.keyword_map.clear()

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

        logger.debug("Split document into %d chunks.", len(chunks))
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

        logger.debug("Split long paragraph into %d sub-chunks.", len(chunks))
        return chunks
