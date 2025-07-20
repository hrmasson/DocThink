import logging
import requests

from application.config import settings
from infrastructure.llm.abstract_llm import ILLMService
from core.models.llm import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class LocalAIMistral(ILLMService):
    """
    LLM service for interacting with a local OpenAI-compatible API (e.g., LocalAI).
    """

    def __init__(self, base_url: str = settings.LOCALAI_URL, model: str = "mistral"):
        self.base_url = base_url
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.model = model
        self.timeout = 300  # seconds
        logger.info(f"LocalAIMistral initialized: endpoint={self.endpoint}, model={self.model}")

    def chat_completion(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [m.model_dump() for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p
        }

        try:
            logger.debug(f"Sending LLM request: {payload}")
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            logger.info(f"LLM response received. Tokens used: {data.get('usage', {}).get('total_tokens')}")
            logger.debug(f"LLM raw response: {data}")

            return LLMResponse(
                text=data["choices"][0]["message"]["content"],
                tokens_used=data["usage"]["total_tokens"],
                is_truncated=False
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            return LLMResponse(
                text=f"LLM error: {str(e)}",
                tokens_used=0,
                is_truncated=True
            )
