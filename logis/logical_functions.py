import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from schemas import LearningResource, ResourceSubject, LearningState, ContentResponse, FeedBack, ContentType
from db.vector_db import build_chroma_db_collection
from sentence_transformers import SentenceTransformer

def retrieve_and_search(state: LearningState) -> dict:
    """
    Retrieve and search for resources based on the current state.
    Returns the top matching resource from the ChromaDB collection.
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


def lesson_decision_node(state: LearningState) -> str:
    """
    Decide lesson style based on curriculum metadata, topic type, and phrasing.
    Returns a string indicating the lesson style.
    """
    topic = state.current_resource.topic.lower()
    unit = state.current_resource.unit.lower()
    desc = state.current_resource.description.lower()

    if "evaluation" in unit:
        style = "evaluation_component"
    elif "practical" in state.current_resource.topic_id or "activity" in topic or "experiment" in desc:
        style = "experimental"
    elif any(keyword in topic for keyword in ["derive", "calculate", "problem", "solve", "formula"]):
        style = "problem_solving"
    elif any(keyword in desc for keyword in ["used in", "applied in", "application", "real-world"]):
        style = "application_based"
    elif "revision" in topic or "summary" in topic:
        style = "revision_summary"
    elif "quiz" in topic or state.content_type == ContentType.QUIZ:
        style = "interactive_quiz"
    elif "enrich" in topic or "context" in desc:
        style = "enrichment"
    else:
        style = "conceptual_focus"

    return style



def parse_chromadb_metadata(metadata: dict) -> LearningResource:
    """
    Convert ChromaDB metadata dict to a LearningResource model.
    Returns a LearningResource instance.
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
    Returns a string indicating the blog style.
    """
    if "importance" in state.current_resource.topic:
        style = "motivational"
    elif state.user.grade >= 12:
        style = "application_focused"
    else:
        style = "storytelling"
    return style

def update_content_count(state: LearningState) -> str:
    """
    Check the content count in the learning state.
    Returns a string indicating if an update is required.
    """
    try:
        if state.count < 4:
            logging.info(f"Current state count: {state.count}")
            return 'Update required'
        else:
            return 'No update required'
    except Exception as e:
        logging.error(f"Error updating state count: {e}")
        return 'No update required'