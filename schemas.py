from enum import Enum
from typing import Optional, Union, List, Dict
from pydantic import BaseModel


class UserInfo(BaseModel):
    username: str
    age: Union[int, float]
    grade: Optional[Union[int, float]] = None


class User(UserInfo):
    id: int
    is_active: bool = True  # Default value

    class Config:
        from_attributes = True


# Better approach for Learning Resource:
class ResourceTopic(str, Enum):
    MATH = "math"
    SCIENCE = "science"
    ENGLISH = "english"
    CHEMISTRY = 'chemistry'


class LearningResource(BaseModel):
    id: int
    topic: ResourceTopic  # Using Enum for fixed options
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
    user: User
    current_resource: Optional[LearningResource] = None
    progress: List[UserProgress] = []

    # Workflow control
    next_action: str = "select_resource"
    history: List[Dict] = []  # For LLM interaction logs