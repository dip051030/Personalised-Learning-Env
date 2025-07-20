from enum import Enum
from typing import Optional, Union, List, Dict
from pydantic import BaseModel, Field


class UserInfo(BaseModel):
    username: str = Field(description="The user's username or handle (e.g., dyane_master).")
    age: Union[int, float, str] = Field(
        description="The user's age. It can be an integer, float, or string (e.g., '22 years old').")
    grade: Optional[Union[int, float, str]] = Field(
        default=None,
        description="The user's grade level, optionally provided. Can be a number (like 12.5) or descriptive string."
    )


class User(UserInfo):
    id: Union[str, int]
    is_active: bool = True

    # user_info: Optional[UserInfo] = None

    class Config:
        from_attributes = True


# Better approach for Learning Resource:
class ResourceTopic(str, Enum):
    MATH = "math"
    SCIENCE = "science"
    ENGLISH = "english"
    CHEMISTRY = 'chemistry'


class LearningResource(BaseModel):
    topic: ResourceTopic
    subtopic: str

    class Config:
        from_attributes = True


class UserProgress(BaseModel):
    id: int
    resource_topic: ResourceTopic
    completed: bool = False

    class Config:
        from_attributes = True


class LearningState(BaseModel):
    # Your existing models
    user: Optional[User] = None
    current_resource: Optional[LearningResource] = None
    progress: List[UserProgress] = []

    # Workflow control
    next_action: str = "select_resource"
    history: List[Dict] = []  # For LLM interaction logs

