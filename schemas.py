from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel


class UserInfo(BaseModel):
    username: str
    age: Union[int, float]  # Better than | for Python < 3.10
    grade: Optional[Union[int, float]] = None  # Clearer typing


class User(UserInfo):
    id: int
    is_active: bool = True  # Default value

    class Config:
        from_attributes = True  # Correct spelling (not 'form_attributes')


# Better approach for Learning Resource:
class ResourceTopic(str, Enum):
    MATH = "math"
    SCIENCE = "science"
    HISTORY = "history"


class LearningResource(BaseModel):
    id: int
    topic: ResourceTopic  # Using Enum for fixed options
    content: str

    class Config:
        from_attributes = True