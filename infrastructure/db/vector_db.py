from abc import ABC, abstractmethod
from typing import List

from core.models.document import Document


class IVectorDatabase(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add a list of documents to the vector database."""
        pass

    @abstractmethod
    def search(self, query: str, filter_roles: List[str], top_k: int = 3) -> List[Document]:
        """Search for documents in the vector database based on a query."""
        pass
