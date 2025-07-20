import requests
from typing import Optional
from core.config import settings
from infrastructure.llm.abstract_llm import ILLMService
from core.models.llm import LLMRequest, LLMResponse

class LocalAIMistral(ILLMService):

    def __init__(self, base_url: str = settings.LOCALAI_URL, model: str = "mistral"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/v1/chat/completions"
        self.model = model
        self.timeout = 60

    def chat_completion(self, request: LLMRequest) -> LLMResponse:
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "messages": [m.model_dump() for m in request.messages],
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            print(response.json())
            return LLMResponse(
                text=response.json()["choices"][0]["message"]["content"],
                tokens_used=response.json()["usage"]["total_tokens"],
                is_truncated=False
            )

        except requests.exceptions.RequestException as e:
            return LLMResponse(
                text=f"LLM error: {str(e)}",
                tokens_used=0,
                is_truncated=True
            )