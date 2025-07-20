import logging
from typing import List, Dict
from core.models.llm import ChatMessage, LLMRequest
from application.config import settings
from core.models.user_query import UserQuery
from application.use_cases.utils import count_tokens

logger = logging.getLogger(__name__)


class RAGUseCase:
    def __init__(self, vector_db, llm_orchestrator):
        self.db = vector_db
        self.llm_orchestrator = llm_orchestrator
        self.max_context_tokens = settings.LLM_MAX_CONTEXT_TOKENS
        self.model_name = settings.LLM_MODEL
        logger.info("RAGUseCase initialized.")

    def execute(self, query: UserQuery) -> Dict:
        logger.info(f"Executing RAG for query: {query.question!r}")
        results = self.db.search(query.question, query.available_roles)

        if not results:
            logger.warning("No relevant chunks found.")
            return {
                "answer": "No relevant information found in the internal knowledge base.",
                "sources": [],
                "is_complete": False
            }

        # Group by source URL
        articles_by_url: Dict[str, List[str]] = {}
        for item in results:
            url = item.get("source_url")
            if url:
                articles_by_url.setdefault(url, []).append(item.get("content", ""))

        if len(articles_by_url) == 1:
            logger.info("Single relevant article identified.")
            source_url = next(iter(articles_by_url))
            chunks = self.db.get_chunks_by_source(source_url)
            answer = self.reason_over_chunks(query.question, chunks)
            return {
                "answer": answer,
                "sources": [source_url],
                "is_complete": True
            }

        logger.info(f"Multiple articles found: {len(articles_by_url)} candidates.")
        response_lines = ["Multiple relevant documents were found:\n"]
        urls = []
        for url, chunks in list(articles_by_url.items())[:5]:
            snippet = chunks[0][:500].strip().replace("\n", " ")
            response_lines.append(f"{url}:\n{snippet}\n")
            urls.append(url)

        response_lines.append("Please clarify your question or follow the links for more details.")
        return {
            "answer": "\n\n".join(response_lines),
            "sources": urls,
            "is_complete": False
        }

    def reason_over_chunks(self, question: str, chunks: List[str]) -> str:
        logger.info("Reasoning over selected chunks...")
        system_tokens = count_tokens(self.llm_orchestrator.system_prompt, self.model_name)
        question_tokens = count_tokens(question, self.model_name)
        base_tokens = system_tokens + question_tokens + 100  # safety buffer

        total_tokens = base_tokens
        selected_chunks: List[str] = []

        for chunk in chunks:
            chunk_text = chunk.strip() + "\n\n"
            chunk_token_count = count_tokens(chunk_text, self.model_name)

            if total_tokens + chunk_token_count > self.max_context_tokens:
                logger.debug(f"Context limit reached at {total_tokens + chunk_token_count} tokens.")
                break

            selected_chunks.append(chunk_text)
            total_tokens += chunk_token_count

        logger.info(f"Using {len(selected_chunks)} chunks ({total_tokens} tokens total).")

        full_context = "".join(selected_chunks)
        logger.debug(f"Combined context:\n{full_context[:1000]}...")  # first 1000 chars

        messages = [
            ChatMessage(role="system", content=self.llm_orchestrator.system_prompt),
            ChatMessage(
                role="user",
                content=(
                    "You are given several parts of internal documentation below.\n"
                    "Answer the question using only that information.\n"
                    "If the answer is not present, respond exactly with:\n"
                    "\"I couldn't find the information in the available context.\"\n\n"
                    f"---\n\n"
                    f"Documentation:\n{full_context}\n"
                    f"---\n\n"
                    f"Question:\n{question}"
                )
            )
        ]

        request = LLMRequest(
            messages=messages,
            max_tokens=settings.LLM_MAX_TOKENS
        )

        logger.info("Sending request to LLM...")
        response = self.llm_orchestrator.get_chat_completion(request)
        logger.info("Received response from LLM.")
        return response
