from typing import Dict

from pydantic import BaseModel


class Document(BaseModel):
    title: str
    content: str
    role: str
    source_url: str


class DocumentChunk(BaseModel):
    content: str
    section_title: str
    metadata: Dict[str, str]
