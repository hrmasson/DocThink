import logging

from application.config import settings
from core.models.llm import ChatMessage, LLMRequest
from infrastructure.llm.abstract_llm import ILLMService

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    def __init__(self, llm_service: ILLMService):
        self.llm = llm_service
        self.system_prompt = (
            "You are an internal AI assistant for company employees. "
            "Your job is to provide accurate, helpful answers based solely on the provided internal documentation. "
            "Do not make assumptions or generate information that is not present in the context. "
            "If the context does not contain the answer, say: \"I couldn't find the information in the available context.\""
        )
        logger.info("LLMOrchestrator initialized with LLM service: %s", type(self.llm).__name__)

    def generate_answer(self, question: str, context: str) -> str:
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=self._build_prompt(question, context))
        ]

        request = LLMRequest(messages=messages, max_tokens=settings.LLM_MAX_TOKENS)
        logger.debug("Sending request to LLM: %s", request)
        response = self.llm.chat_completion(request)
        logger.debug("LLM response received.")
        return response.text

    def get_chat_completion(self, request: LLMRequest) -> str:
        logger.debug("get_chat_completion called. Max tokens: %s", request.max_tokens)
        response = self.llm.chat_completion(request)
        return response.text

    def _build_prompt(self, question: str, context: str) -> str:
        promt = (
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            "Formulate the exact response based on the context. "
            "If there is no answer, say \"I can't find information.\""
        )
        logger.debug("Prompt built for LLM: %s", promt[:200])
        return promt
