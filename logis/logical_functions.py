import logging

import chromadb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from schemas import LearningResource, ResourceSubject, LearningState, ContentType
from db.vector_db import build_chroma_db_collection, save_scraped_data_to_vdb
from sentence_transformers import SentenceTransformer


def load_or_build_collections(vdb_path, lessons_collection, scraped_collection):
    client = chromadb.PersistentClient(path=vdb_path)

    try:
        lessons_col = client.get_collection(lessons_collection)
        logging.info(f"Collection '{lessons_collection}' loaded successfully.")
    except Exception as e:
        logging.warning(f"Collection '{lessons_collection}' not found. Building it now.")
        build_chroma_db_collection()
        lessons_col = client.get_collection(lessons_collection)

    try:
        scraped_col = client.get_collection(scraped_collection)
        logging.info(f"Collection '{scraped_collection}' loaded successfully.")
    except Exception as e:
        logging.warning(f"Collection '{scraped_collection}' not found. Building it now.")
        save_scraped_data_to_vdb()
        scraped_col = client.get_collection(scraped_collection)

    return lessons_col, scraped_col


def search_both_collections(state: LearningState,
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
            logging.warning(
                f"[logical_functions.py:{search_both_collections.__code__.co_firstlineno}] WARNING No current_resource in state.")
            return None

        lessons_col, scraped_col = load_or_build_collections(vdb_path, lessons_collection, scraped_collection)

        # Load the embedding model
        model = SentenceTransformer("Shashwat13333/bge-base-en-v1.5_v4")
        query_text = state.current_resource.topic
        query_embedding = model.encode(query_text).tolist()

        # Query both collections
        lessons_results = lessons_col.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        scraped_results = scraped_col.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )

        logging.info(
            f"[logical_functions.py:{search_both_collections.__code__.co_firstlineno}] INFO Queried both collections for topic '{query_text}'.")
        return {
            "lessons_results": lessons_results,
            "scraped_results": scraped_results
        }
    except Exception as e:
        logging.error(
            f"[logical_functions.py:{search_both_collections.__code__.co_firstlineno}] ERROR Error searching collections: {e}")
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
            logging.info(
                f"[logical_functions.py:{update_content_count.__code__.co_firstlineno}] INFO Current state count: {state.count}")
            return 'Update required'
        else:
            logging.info(
                f"[logical_functions.py:{update_content_count.__code__.co_firstlineno}] INFO No update required, current count: {state.count}")
            return 'No update required'
    except Exception as e:
        logging.error(
            f"[logical_functions.py:{update_content_count.__code__.co_firstlineno}] ERROR Error updating state count: {e}")
        return 'No update required'
