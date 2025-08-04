from enum import Enum
from typing import Optional, Union, List, Dict
from pydantic import BaseModel, Field, HttpUrl
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

class WebCrawlerConfig(BaseModel):
    url: HttpUrl = Field(..., description="Original source URL of the educational web page.")
    title: Optional[str] = Field(None, description="Main H1 title of the page. Must be clearly visible and educational. Return null if not found.")
    headings: List[str] = Field(default_factory=list, description="Ordered list of H2–H4 section or sub-section headings from the main article body only.")
    main_findings: List[str] = Field(default_factory=list, description="List of core educational statements (facts, definitions, laws, etc.) extracted from visible content. No hallucinated or inferred info.")
    content: Optional[str] = Field(None, description="Full combined educational content as one plain block, preserving original phrasing and order. Join all main_findings.")
    word_count: int = Field(default=0, description="Word count of the `content` field. Must be >= 50 to be considered useful.")


class LearningState(BaseModel):
    user: UserInfo
    current_resource: Optional[LearningResource] = None
    enriched_resource: Optional[EnrichedLearningResource] = None
    progress: List[UserProgress] = []
    topic_data: Optional[dict] = None
    related_examples: Optional[List[str]] = None
    content_type: ContentType = ContentType.LESSON
    content: Optional[ContentResponse] = None
    next_action: Optional[RouteSelector] = Field(default="lesson_selection", description="Should return lesson_selection or blog_selection")
    history: List[HistoryEntry] = []
    feedback: Optional[FeedBack] = None
    count: int = 0

    class Config:
        from_attributes = True