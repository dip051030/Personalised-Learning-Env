"""schemas.py

This module defines the Pydantic models used for data validation and structuring
throughout the personalized learning system. These schemas ensure data consistency
and provide clear definitions for various entities like user information,
learning resources, content responses, and the overall learning state.
"""
from datetime import datetime
from enum import Enum
from typing import List, Optional, Union, Dict, Any

from pydantic import BaseModel, Field, HttpUrl


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
    age: str
    grade: Optional[str] = None
    id: str
    is_active: bool = True
    user_info: Optional[str] = Field(default='', description="Summarised User Info. ")

    class Config:
        from_attributes = True


class LearningResource(BaseModel):
    subject: ResourceSubject
    grade: int
    unit: Optional[str] = None
    topic_id: Optional[str] = None
    topic: str
    description: Optional[str] = None
    elaboration: Optional[str] = None
    keywords: List[str] = []
    hours: int
    references: Optional[str] = None

    class Config:
        from_attributes = True


class UserProgress(BaseModel):
    id: int
    user_id: str
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
    user_id: str
    resource: LearningResource
    timestamp: datetime
    action: str

    class Config:
        from_attributes = True


class EnrichedLearningResource(BaseModel):
    subject: ResourceSubject
    grade: int
    unit: Optional[str] = None
    topic_id: Optional[str] = None
    topic: str
    description: Optional[str] = None
    elaboration: Optional[str] = None
    keywords: List[str] = []
    hours: int
    references: Optional[str] = None


class FeedBack(BaseModel):
    rating: int = 1
    comments: Optional[str] = None
    needed: bool = Field(default=True, description="True if feedback is needed else False")
    gaps: Optional[List[str]] = Field(default=[], description="List of gaps in the content that need to be addressed")
    ai_reliability_score: Optional[float] = Field(default=0.0,
                                                  description="AI reliability score for the content, between 0 and 1")

    class Config:
        from_attributes = True


class RouteSelector(BaseModel):
    next_node: str


class WebCrawlerConfig(BaseModel):
    url: HttpUrl = Field(..., description="Original URL of the scraped educational page.")
    source: str = Field(..., description="Domain source of the page, e.g., 'byjus.com'.")
    content_type: str = Field(..., description="Type of content on the page, e.g., youtube video.")

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


class PostValidationResult(BaseModel):
    is_valid: bool = Field(default=False,
                           description="Indicates whether the SEO blog post passed all validation checks.")
    violations: List[str] = Field(default_factory=list,
                                  description="List of descriptive violation messages if validation failed. Empty if valid.")


class HistoryItem(BaseModel):
    topic: str
    content: str
    feedback: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class LearningState(BaseModel):
    user: Optional[UserInfo] = None
    current_resource: Optional[LearningResource] = None
    progress: List[str] = []
    next_action: Optional[RouteSelector] = None
    history: List[HistoryItem] = []
    enriched_resource: Optional[EnrichedLearningResource] = None
    topic_data: Optional[List[Dict[str, Any]]] = None
    related_examples: Optional[List[str]] = None
    content_type: ContentType = ContentType.LESSON
    content: Optional[ContentResponse] = None
    feedback: Optional[FeedBack] = None
    validation_result: Optional[PostValidationResult] = None
    count: int = 0

