import logging
import chromadb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from schemas import LearningResource, ResourceSubject, LearningState, ContentResponse, FeedBack, ContentType
from db.vector_db import build_chroma_db_collection, save_scraped_data_to_vdb
from sentence_transformers import SentenceTransformer


def search_both_collections(state : LearningState,
                            vdb_path="./local VDB/chromadb",
                            lessons_collection="lessons",
                            scraped_collection="scraped_data",
                            n_results=1):
    """
    Search both the lessons and scraped_data collections for the most similar items to the query.
    Returns results from both collections.
    """
    try:
        if state.current_resource is None:
            return None


        build_chroma_db_collection()
        save_scraped_data_to_vdb()
        # Load the embedding model
        model = SentenceTransformer("Shashwat13333/bge-base-en-v1.5_v4")
        query_text = state.current_resource.topic
        query_embedding = model.encode(query_text).tolist()

        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=vdb_path)
        lessons_col = client.get_or_create_collection(lessons_collection)
        scraped_col = client.get_or_create_collection(scraped_collection)

        # Query both collections
        lessons_results = lessons_col.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        scraped_results = scraped_col.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        return {
            "lessons_results": lessons_results,
            "scraped_results": scraped_results
        }
    except Exception as e:
        logging.error(f"Error searching collections: {e}")
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