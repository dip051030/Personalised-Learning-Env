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
    id: Union[str, int]
    is_active: bool = True

    class Config:
        from_attributes = True


# Better approach for Learning Resource:
class ResourceTopic(str, Enum):
    MATH = "math"
    SCIENCE = "science"
    ENGLISH = "english"
    CHEMISTRY = 'chemistry'
    PHYSICS = 'physics'


class LearningResource(BaseModel):
    topic: ResourceTopic
    subtopic: str

    class Config:
        from_attributes = True


class UserProgress(BaseModel):
    id: int
    resource_topic: LearningResource
    completed: bool = False

    class Config:
        from_attributes = True


class ContentResponse(BaseModel):
    content: str = Field(description="The generated content from the LLM.")
    metadata: Optional[dict] = Field(default=None, description="Optional metadata about the content generation.")

    class Config:
        from_attributes = True

class HistoryEntry(BaseModel):
    user_info: dict
    resource_data: dict
    generated_content: ContentResponse

class LearningState(BaseModel):
    # Your existing models
    user: UserInfo
    current_resource: Optional[LearningResource] = None
    progress: List[UserProgress] = []

    # Workflow control
    next_action: str = "select_resource"
    history: List[HistoryEntry] = []

