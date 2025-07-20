import logging

from fastapi import APIRouter, Depends

from application.use_cases.rag import RAGUseCase
from core.models.answer import AnswerResponse
from core.models.user_query import UserQuery
from dependencies import get_rag_use_case

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(
        query: UserQuery,
        use_case: RAGUseCase = Depends(get_rag_use_case)
) -> AnswerResponse:
    logger.info(f"Received query: '{query.question}' with roles: {query.available_roles}")
    result = use_case.execute(query)
    logger.info(f"Generated answer. Complete: {result['is_complete']}. Sources: {result['sources']}")
    return AnswerResponse(**result)
