import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from schemas import LearningResource, ResourceSubject, LearningState, ContentResponse, FeedBack
from db.vector_db import build_chroma_db_collection
from sentence_transformers import SentenceTransformer

def retrieve_and_search(state: LearningState) -> dict:
    """
    Retrieve and search for resources based on the current state.
    """
    try:
        if state.current_resource is not None:
            collection, model = build_chroma_db_collection('class_11_physics.json', collection_name='lessons')
            query_embedding = model.encode([state.current_resource.topic]).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=1
            )
            return results
    except Exception as e:
        logging.error(f"Error retrieving and searching resources: {e}")
        return None

# def decision_node(state: LearningState) -> str:
#     """
#     Decide whether to generate a lesson or a blog.
#     """
#     topic = state.current_resource.topic
#     grade = int(state.user.grade)
#     blog_keywords = []
#     lesson_keywords = []
#
#     if any(kw in topic for kw in blog_keywords) and grade > 10:
#         return "blog"
#     elif any(kw in topic for kw in lesson_keywords):
#         return "lesson"
#     elif grade <= 10:
#         return "lesson"
#     else:
#         return "lesson"

def lesson_decision_node(state: LearningState) -> str:
    """
    Decide lesson style based on user grade and topic.
    """
    if "practice" in state.current_resource.topic:
        style = "exercise_heavy"
    else:
        style = "general_concept"
    return style

def parse_chromadb_metadata(metadata: dict) -> LearningResource:
    """
    Convert ChromaDB metadata dict to a LearningResource model.
    """
    return LearningResource(
    subject=ResourceSubject(metadata.get('subject', 'unknown').lower()),
    grade=metadata.get("grade"),
    unit=metadata.get("unit"),
    topic_id=metadata.get("topic_id"),
    topic=metadata.get("topic_title"),
    description=metadata.get("description", ""),
    keywords=metadata.get("keywords").split(","),
    hours=metadata.get("hours"),
    references=metadata.get("references"),
    elaboration=metadata.get("elaboration", "")
)

def blog_decision_node(state: LearningState) -> str:
    """
    Decide blog style based on topic and user grade.
    """
    if "importance" in state.current_resource.topic:
        style = "motivational"
    elif state.user.grade >= 12:
        style = "application_focused"
    else:
        style = "storytelling"
    return style