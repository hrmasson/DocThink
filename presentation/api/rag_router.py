from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from core.models.user_query import UserQuery
from application.use_cases.rag import RAGUseCase
from dependencies import get_rag_use_case

router = APIRouter()

@router.post("/ask")
async def ask_question(query: UserQuery,
                       use_case: RAGUseCase = Depends(get_rag_use_case)):
    answer, sources = use_case.execute(query)
    return {
        "answer": answer,
        "sources": sources
    }