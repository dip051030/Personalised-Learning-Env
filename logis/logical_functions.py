import logging
from schemas import LearningResource, ResourceSubject, LearningState
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

def decision_node(state: LearningState) -> str:
    """
    Decide whether to generate a lesson or a blog.
    """
    topic = state.current_resource.topic
    grade = int(state.user.grade)
    blog_keywords = []
    lesson_keywords = []

    if any(kw in topic for kw in blog_keywords) and grade > 10:
        return "blog"
    elif any(kw in topic for kw in lesson_keywords):
        return "lesson"
    elif grade <= 10:
        return "lesson"
    else:
        return "lesson"

def lesson_decision_node(state: LearningState) -> str:
    """
    Decide lesson style based on user grade and topic.
    """
    if state.user.grade <= 6:
        style = "kid_friendly"
    elif state.user.grade >= 10:
        style = "exam_ready"
    elif "practice" in state.current_resource.topic:
        style = "exercise_heavy"
    else:
        style = "general_concept"
    return style

def parse_chromadb_metadata(metadata: dict) -> LearningResource:
    """
    Convert ChromaDB metadata dict to a LearningResource model.
    """
    return LearningResource(
        subject=ResourceSubject(metadata["subject"].lower()),
        grade=metadata["grade"],
        unit=metadata["unit"],
        topic_id=metadata["topic_id"],
        topic=metadata["topic_title"],
        description=metadata["description"],
        elaboration=metadata.get("elaboration"),
        keywords=metadata["keywords"],
        hours=metadata["hours"],
        references=metadata["references"]
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