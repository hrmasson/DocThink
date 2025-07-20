import json
import logging
from pathlib import Path
from typing import List

from tqdm import tqdm

from application.config import settings
from core.models.document import Document
from infrastructure.db.chroma_db import ChromaDB

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_documents(json_dir: Path) -> List[Document]:
    """Load and parse JSON documents from a given directory."""
    logger.info(f"Loading documents from directory: {json_dir}")
    documents = []

    for json_file in json_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents.append(Document(**data))
            logger.info(f"Loaded file: {json_file.name}")
        except Exception as e:
            logger.error(f"Failed to load {json_file.name}: {e}")

    return documents


def main():
    # Check if the source directory exists
    json_dir = Path(settings.PARSED_DOCS_DIR)
    if not json_dir.exists():
        logger.error(f"Directory not found: {json_dir}")
        return

    # Initialize ChromaDB
    try:
        db = ChromaDB()
        logger.info(f"ChromaDB initialized at: {settings.CHROMADB_DIR}")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        return

    # Load documents
    documents = load_documents(json_dir)
    if not documents:
        logger.warning("No valid JSON documents found to index.")
        return

    # Index documents
    logger.info(f"Starting indexing of {len(documents)} documents...")
    for doc in tqdm(documents, desc="Indexing documents"):
        try:
            db.add_documents([doc])
        except Exception as e:
            logger.error(f"Failed to index document '{doc.title}': {e}")

    logger.info("Indexing completed successfully.")


if __name__ == "__main__":
    main()
