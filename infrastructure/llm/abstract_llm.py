from abc import ABC, abstractmethod
from core.models.llm import LLMRequest, LLMResponse

class ILLMService(ABC):

    @abstractmethod
    def chat_completion(self, request: LLMRequest) -> LLMResponse:
        pass