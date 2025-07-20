from typing import List

from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

class LLMRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class LLMResponse(BaseModel):
    text: str
    tokens_used: int
    is_truncated: bool
