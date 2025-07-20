import json
from pathlib import Path
from typing import List
from tqdm import tqdm
import logging
from core.config import settings
from core.models.document import Document
from infrastructure.db.chroma_db import ChromaDB

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_documents(json_dir: Path) -> List[Document]:
    print(f"Loading documents from {json_dir}")
    """Загружает JSON-документы из указанной директории"""
    documents = []
    for json_file in json_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                documents.append(Document(**data))
            logger.info(f"Успешно загружен: {json_file.name}")
        except Exception as e:
            logger.error(f"Ошибка загрузки {json_file}: {str(e)}")
    return documents

def main():
    # Проверка существования директории
    json_dir = Path(settings.PARSED_DOCS_DIR)
    if not json_dir.exists():
        logger.error(f"Директория не найдена: {json_dir}")
        return

    # Инициализация ChromaDB
    try:
        db = ChromaDB()
        logger.info(f"База данных инициализирована в {settings.CHROMADB_DIR}")
    except Exception as e:
        logger.error(f"Ошибка инициализации базы данных: {str(e)}")
        return

    # Загрузка и индексация документов
    documents = load_documents(json_dir)
    if not documents:
        logger.warning("Не найдено подходящих JSON-документов")
        return

    logger.info(f"Начата индексация {len(documents)} документов...")
    for doc in tqdm(documents, desc="Обработка документов"):
        try:
            db.add_documents([doc])
        except Exception as e:
            logger.error(f"Ошибка индексации документа {doc.title}: {str(e)}")

    logger.info("Индексация успешно завершена")


if __name__ == "__main__":
    main()