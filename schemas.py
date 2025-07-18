from enum import Enum
from typing import TypedDict, Optional
from pydantic import BaseModel


class UserInfo(BaseModel):
    username: str
    age:  int | float
    grade: Optional[int | float] = None

class User(UserInfo):
    id: int
    is_active: bool

    class Config:
        from_attributes = True

class CurrentLearningResource(str, Enum):
