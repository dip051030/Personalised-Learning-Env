from enum import Enum
from typing import List, Optional, Union
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
    ai_reliability_score : Optional[float] = Field(default=0.0, description="AI reliability score for the content, between 0 and 1")

    class Config:
        from_attributes = True


class RouteSelector(BaseModel):
    next_node: str

class WebCrawlerConfig(BaseModel):
    url: HttpUrl = Field(..., description="Original URL of the scraped educational page.")
    source: str = Field(..., description="Domain source of the page, e.g., 'byjus.com'.")

    subject: str = Field(..., description="Aligned school subject (e.g., Physics, Chemistry).")
    grade: int = Field(..., description="Grade level the content is aligned to (e.g., 11).")
    unit: str = Field(..., description="Curriculum unit title (e.g., 'Electricity and Magnetism').")
    topic_title: Optional[str] = Field(None, description="Optional topic title if entity matched.")

    title: Optional[str] = Field(None, description="Main H1 title from the article body.")
    headings: List[str] = Field(default_factory=list, description="H2–H4 headings extracted from body.")
    main_findings: List[str] = Field(default_factory=list, description="Key factual or conceptual findings.")
    content: Optional[str] = Field(None, description="Full clean text block, joined from findings or body.")
    keywords: List[str] = Field(default_factory=list, description="Extracted or matched educational keywords.")

    word_count: int = Field(ge=0, default=0, description="Total word count of the `content`.")
    status: str = Field(..., description="Status of the scrape: success or failed.")
    scraped_at: datetime = Field(..., description="Timestamp when this page was scraped.")




class LearningState(BaseModel):
    user: UserInfo
    current_resource: Optional[LearningResource] = None
    enriched_resource: Optional[EnrichedLearningResource] = None
    progress: List[UserProgress] = []
    topic_data: Optional[list] = None
    related_examples: Optional[List[str]] = None
    content_type: ContentType = ContentType.LESSON
    content: Optional[ContentResponse] = None
    next_action: Optional[RouteSelector] = Field(default="lesson_selection", description="Should return lesson_selection or blog_selection")
    history: List[HistoryEntry] = []
    feedback: Optional[FeedBack] = None
    count: int = 0

    class Config:
        from_attributes = True