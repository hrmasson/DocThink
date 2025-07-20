import logging
from abc import ABC, abstractmethod
from typing import List

from core.models.document import Document

logger = logging.getLogger(__name__)


class IVectorDatabase(ABC):
    """
    Interface for a vector database used in the RAG system.
    Provides methods for adding documents, searching, and retrieving chunk data.
    """

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add a list of documents (with content) to the vector database.

        Args:
            documents (List[Document]): Documents to be embedded and indexed.
        """
        pass

    @abstractmethod
    def search(self, query: str, filter_roles: List[str], top_k: int = 3) -> List[Document]:
        """
        Perform a search in the vector database using a query and optional role filter.

        Args:
            query (str): User input or search query.
            filter_roles (List[str]): List of user roles for filtering.
            top_k (int): Max number of results to return (default: 3).

        Returns:
            List[Document]: Ranked list of relevant document chunks.
        """
        pass

    @abstractmethod
    def get_chunks_by_source(self, source_url: str) -> List[str]:
        """
        Get document chunks associated with a specific source URL.

        Args:
            source_url (str): Original source identifier.

        Returns:
            List[str]: Ordered list of document chunk texts.
        """
        pass
