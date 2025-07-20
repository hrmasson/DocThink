import logging

from application.services.llm_orchestrator import LLMOrchestrator
from application.use_cases.rag import RAGUseCase
from application.config import settings
from infrastructure.db.chroma_db import ChromaDB
from infrastructure.llm.localai_mistral import LocalAIMistral

logger = logging.getLogger(__name__)

logger.info("Initializing vector DB")
vector_db_instance = ChromaDB()

logger.info("Initializing LLM instance")
llm_instance = LocalAIMistral(
    base_url=settings.LOCALAI_URL,
    model=settings.LLM_MODEL
)

logger.info("Initializing RAG use case")
rag_use_case_instance = RAGUseCase(vector_db_instance, LLMOrchestrator(llm_instance))


def get_vector_db():
    return vector_db_instance


def get_llm_service():
    return llm_instance


def get_rag_use_case():
    return rag_use_case_instance


logger.info("All core services initialized")
