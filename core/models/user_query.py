from typing import List

from pydantic import BaseModel


class UserQuery(BaseModel):
    question: str
    available_roles: List[str]
