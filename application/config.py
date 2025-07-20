from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    DOC_ROLES: list[str] = ["admin", "developer", "jurist"]

    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    CHROMADB_DIR = DATA_DIR / "chroma_db"
    PARSED_DOCS_DIR = DATA_DIR / "parsed_docs"
    KEYWORDS_FILE = DATA_DIR / "keywords" / "keyword_map.json"

    # LocalAI
    LOCALAI_URL = "http://localhost:8083"
    LLM_MODEL = "mistral"

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MAX_CONTEXT_TOKENS = 2200  # input limit
    LLM_MAX_TOKENS = 512  # output limit

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # API
    API_HOST = "0.0.0.0"
    API_PORT = 8084
    DEBUG = True


settings = Settings()
