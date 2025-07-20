from application.services.llm_orchestrator import LLMOrchestrator
from core.models.user_query import UserQuery


class RAGUseCase:
    def __init__(self, vector_db, llm_service):
        self.db = vector_db
        self.llm_orchestrator = LLMOrchestrator(llm_service)

    def execute(self, query: UserQuery) -> str:
        contexts = self.db.search(query.question, query.available_roles)
        print(f"contexts {contexts}")
        return self.llm_orchestrator.generate_answer(query.question, "\n".join(contexts))