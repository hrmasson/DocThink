from fastapi import Depends

from infrastructure.db.chroma_db import ChromaDB
from infrastructure.llm.localai_mistral import LocalAIMistral
from application.services.llm_orchestrator import LLMOrchestrator
from application.use_cases.rag import RAGUseCase
from core.config import settings

def get_vector_db():
    return ChromaDB()

def get_llm_service():
    return LocalAIMistral(
        base_url=settings.LOCALAI_URL,
        model=settings.LLM_MODEL
    )

def get_rag_use_case(db: ChromaDB = Depends(get_vector_db),
                     llm: LocalAIMistral = Depends(get_llm_service)):
    return RAGUseCase(db, LLMOrchestrator(llm))