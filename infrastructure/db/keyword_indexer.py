from keybert import KeyBERT
from typing import List, Dict, Tuple

from core.models.document import DocumentChunk


class KeywordIndexer:
    def __init__(self):
        self.model = KeyBERT()
        self.keyword_map: Dict[str, List[str]] = {}

    def extract_keywords(
            self,
            text: str,
            top_n: int = 5,
            min_confidence: float = 0.25
    ) -> List[Tuple[str, float]]:
        """
        Extracts and filters keyphrases from text with confidence scores.
        """
        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n * 3
        )

        filtered_keywords = [
            (kw[0].lower(), kw[1])
            for kw in keywords
            if kw[1] >= min_confidence
        ]
        filtered_keywords.sort(key=lambda x: x[1], reverse=True)

        return filtered_keywords[:top_n]

    def search(self, query: str) -> List[str]:
        """
        Returns list of keywords from the query that are present in the index.
        """
        extracted = self.extract_keywords(query, top_n=10, min_confidence=0.25)
        return [kw for kw, _ in extracted if kw in self.keyword_map]

    def index_keywords(
            self,
            chunk: DocumentChunk,
            chunk_id: str,
            top_n: int = 10,
            min_confidence: float = 0.25
    ):
        """
        Indexes keywords from a single chunk by mapping them to the chunk_id.
        """
        keywords = self.extract_keywords(chunk.content, top_n, min_confidence)
        for kw, _ in keywords:
            self.keyword_map.setdefault(kw, []).append(chunk_id)
