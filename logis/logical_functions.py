import logging

import chromadb

from db.vector_db import build_chroma_db_collection, save_scraped_data_to_vdb
from models.embedding_model import embedding_model
from schemas import LearningResource, ResourceSubject, LearningState, ContentType


def load_or_build_collections(username: str, vdb_path: str):
    lessons_collection = f"{username}_lessons"
    scraped_collection = f"{username}_scraped_data"
    client = chromadb.PersistentClient(path=vdb_path)

    try:
        lessons_col = client.get_collection(lessons_collection)
        logging.info(f"Collection '{lessons_collection}' loaded successfully.")
    except Exception:
        logging.warning(f"Collection '{lessons_collection}' not found. Building it now.")
        build_chroma_db_collection(username=username)
        lessons_col = client.get_collection(lessons_collection)

    try:
        scraped_col = client.get_collection(scraped_collection)
        logging.info(f"Collection '{scraped_collection}' loaded successfully.")
    except Exception:
        logging.warning(f"Collection '{scraped_collection}' not found. Building it now.")
        save_scraped_data_to_vdb(username=username)
        scraped_col = client.get_collection(scraped_collection)

    return lessons_col, scraped_col


def search_both_collections(state: LearningState,
                            vdb_path="/tmp/local VDB/chromadb",
                            n_results=1):
    try:
        if state.current_resource is None:
            logging.warning("WARNING No current_resource in state.")
            return None
        
        if state.user is None:
            logging.warning("WARNING No user in state.")
            return None
            
        username = state.user.username
        lessons_col, scraped_col = load_or_build_collections(username, vdb_path)

        query_text = state.current_resource.topic
        try:
            query_embedding = embedding_model.encode(query_text).tolist()
        except Exception as e:
            logging.error(f"Error encoding query text for embedding: {e}")
            return None

        try:
            lessons_results = lessons_col.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
        except chromadb.exceptions.ChromaDBException as e:
            logging.error(f"Error querying lessons collection: {e}")
            lessons_results = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

        try:
            scraped_results = scraped_col.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
        except chromadb.exceptions.ChromaDBException as e:
            logging.error(f"Error querying scraped collection: {e}")
            scraped_results = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

        logging.info(
            f"INFO Queried both collections for topic '{query_text}'.")
        return {
            "lessons_results": lessons_results,
            "scraped_results": scraped_results
        }
    except Exception as e:
        logging.error(
            f"ERROR Error searching collections: {e}")
        return None


def lesson_decision_node(state: LearningState) -> str:
    topic = state.current_resource.topic.lower() if state.current_resource.topic else ""
    unit = state.current_resource.unit.lower() if state.current_resource.unit else ""
    desc = state.current_resource.description.lower() if state.current_resource.description else ""

    if "evaluation" in unit:
        style = "evaluation_component"
    elif "practical" in (state.current_resource.topic_id or "") or "activity" in topic or "experiment" in desc:
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
    if state.current_resource and "importance" in state.current_resource.topic:
        style = "motivational"
    elif state.user and int(state.user.grade) >= 12:
        style = "application_focused"
    else:
        style = "storytelling"
    return style


def update_content_count(state: LearningState) -> str:
    try:
        if state.count < 4:
            logging.info(f"INFO Current state count: {state.count}")
            return 'Update required'
        else:
            logging.info(f"INFO No update required, current count: {state.count}")
            return 'No update required'
    except Exception as e:
        logging.error(f"ERROR Error updating state count: {e}")
        return 'No update required'
