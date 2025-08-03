from enum import Enum
from typing import Optional, Union, List, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class ResourceSubject(str, Enum):
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    MATH = "math"
    ENGLISH = "english"
    SCIENCE = "science"


class ContentType(str, Enum):
    LESSON = "lesson"
    QUIZ = "quiz"
    PROJECT = "project"
    PRACTICAL = "practical"


class UserInfo(BaseModel):
    username: str
    age: Union[int, float, str]
    grade: Optional[Union[int, float, str]] = None
    id: Union[str, int]
    is_active: bool = True
    user_info: Optional[str] = Field(default='', description="Summarised User Info. ")

    class Config:
        from_attributes = True


class LearningResource(BaseModel):
    subject: ResourceSubject
    grade: int
    unit: str
    topic_id: str
    topic: str
    description: str
    elaboration: Optional[str] = None
    keywords: List[str] = []
    hours: int
    references: str

    class Config:
        from_attributes = True


class UserProgress(BaseModel):
    id: int
    user_id: Union[str, int]
    resource: LearningResource
    completed: bool = False
    completion_date: Optional[datetime] = None
    score: Optional[float] = None

    class Config:
        from_attributes = True


class ContentResponse(BaseModel):
    content: str

    class Config:
        from_attributes = True


class HistoryEntry(BaseModel):
    user_id: Union[str, int]
    resource: LearningResource
    timestamp: datetime
    action: str

    class Config:
        from_attributes = True


class EnrichedLearningResource(BaseModel):
    subject: ResourceSubject
    grade: int
    unit: str
    topic_id: str
    topic: str
    description: str
    elaboration: Optional[str] = None
    keywords: List[str] = []
    hours: int
    references: str


class FeedBack(BaseModel):
    rating: int = 1
    comments: Optional[str] = None
    needed: bool = Field(default=True, description="True if feedback is needed else False")
    gaps: Optional[List[str]] = Field(default=[], description="List of gaps in the content that need to be addressed")

    class Config:
        from_attributes = True


class RouteSelector(BaseModel):
    next_node: str


class LearningState(BaseModel):
    user: UserInfo
    current_resource: Optional[LearningResource] = None
    enriched_resource: Optional[EnrichedLearningResource] = None
    progress: List[UserProgress] = []
    topic_data: Optional[Dict] = None
    related_examples: Optional[List[str]] = None
    content_type: ContentType = ContentType.LESSON
    content: Optional[ContentResponse] = None
    next_action: Optional[RouteSelector] = Field(default="lesson_selection", description="Should return lesson_selection or blog_selection")
    history: List[HistoryEntry] = []
    feedback: Optional[FeedBack] = None
    count: int = 0

    class Config:
        from_attributes = True
