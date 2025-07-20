import logging
from abc import ABC, abstractmethod

from core.models.llm import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class ILLMService(ABC):
    """
    Interface for communication with a Language Model (LLM) service.
    Defines the contract for sending chat-based completion requests.
    """

    @abstractmethod
    def chat_completion(self, request: LLMRequest) -> LLMResponse:
        """
        Send a chat completion request to the LLM service.

        Args:
            request (LLMRequest): The request object containing messages, tokens, and parameters.

        Returns:
            LLMResponse: The response object containing the generated text and metadata.
        """
        pass
