from core.models.llm import ChatMessage, LLMRequest, LLMResponse
from infrastructure.llm.abstract_llm import ILLMService
from core.config import settings


class LLMOrchestrator:
    def __init__(self, llm_service: ILLMService):
        self.llm = llm_service
        self.system_prompt = (
            "You are an AI assistant for Bitbucket developers within the company. "
            "Your job is to provide clear, accurate answers to technical questions using only the provided internal knowledge base. "
            "Do not make assumptions or fabricate information. "
            "If the answer is not in the context, reply with: \"I couldn't find the information in the available context.\""
        )
    def generate_answer(self, question: str, context: str) -> str:
        messages = [
            ChatMessage(role="system", content=self.system_prompt),
            ChatMessage(role="user", content=self._build_prompt(question, context))
        ]

        response = self.llm.chat_completion(
            LLMRequest(
                messages=messages,
                max_tokens=settings.LLM_MAX_TOKENS
            )
        )
        return response.text

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            "Formulate the exact response based on the context. "
            "If there is no answer, say \"I can't find information.\""
        )
