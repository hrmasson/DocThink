from core.models.llm import LLMRequest, ChatMessage
from infrastructure.llm.local_mistral import LocalAIMistral

llm = LocalAIMistral(base_url="http://localhost:8083")
system_prompt = """Ты - ассистент для разработчиков
Отвечай точно на технические вопросы"""
messages = [
    ChatMessage(role="system", content=system_prompt),
    ChatMessage(role="user", content="что такое ноутбук?")
]

request = LLMRequest(
        messages=messages
    )



response = llm.generate(request)
print(response.text, response.tokens_used, response.is_truncated)