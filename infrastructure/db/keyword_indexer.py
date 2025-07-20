import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

from keybert import KeyBERT
from core.models.document import DocumentChunk

logger = logging.getLogger(__name__)


class KeywordIndexer:
    def __init__(self, cache_path: Path = Path("keyword_map.json")):
        self.model = KeyBERT()
        self.keyword_map: Dict[str, List[str]] = {}
        self.cache_path = cache_path
        self._load_cache()

    def extract_keywords(
            self,
            text: str,
            top_n: int = 10,
            min_confidence: float = 0.1
    ) -> List[Tuple[str, float]]:
        raw_keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n * 3
        )

        filtered = [
            (kw.lower(), score)
            for kw, score in raw_keywords
            if score >= min_confidence and len(kw.split()) > 1
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:top_n]

    def search(self, query: str, top_n: int = 10, min_confidence: float = 0.1) -> List[str]:
        extracted = self.extract_keywords(query, top_n, min_confidence)
        keywords = [kw for kw, _ in extracted]
        logger.debug("Extracted keywords from query '%s': %s", query, keywords)

        matched_keywords = set()

        for kw in keywords:
            if kw in self.keyword_map:
                matched_keywords.add(kw)
            else:
                for word in kw.split():
                    if word in self.keyword_map:
                        matched_keywords.add(word)

        return list(matched_keywords)

    def index_keywords(
            self,
            chunk: DocumentChunk,
            chunk_id: str,
            top_n: int = 10,
            min_confidence: float = 0.1
    ):
        keywords = self.extract_keywords(chunk.content, top_n, min_confidence)
        logger.debug("Indexing chunk %s with keywords: %s", chunk_id, [kw for kw, _ in keywords])
        for kw, _ in keywords:
            self.keyword_map.setdefault(kw, []).append(chunk_id)

    def save_cache(self):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.keyword_map, f, ensure_ascii=False, indent=2)
            logger.info("Saved keyword_map with %d keywords to '%s'", len(self.keyword_map), self.cache_path)
        except Exception as e:
            logger.error("Failed to save keyword_map to '%s': %s", self.cache_path, e)

    def _load_cache(self):
        if not self.cache_path.exists():
            logger.info("No existing keyword_map cache found at '%s'. Starting fresh.", self.cache_path)
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.keyword_map = json.load(f)
            logger.info("Loaded keyword_map with %d keywords from '%s'", len(self.keyword_map), self.cache_path)
        except Exception as e:
            logger.error("Failed to load keyword_map from '%s': %s", self.cache_path, e)
