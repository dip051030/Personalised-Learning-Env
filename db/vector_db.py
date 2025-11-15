"""
Vector database utilities for building and saving ChromaDB collections from lesson and scraped data.
"""
import logging

import chromadb

from db.loader import load_json_data
from models.embedding_model import embedding_model


def sanitize_metadata(metadata: dict) -> dict:
    return {k: (",".join(v) if isinstance(v, list) and v is not None else str(v) if v is not None else '') 
            for k, v in metadata.items()}


def clean_metadata(metadata: dict) -> dict:
    return {k: v for k, v in metadata.items() if v is not None}


def build_chroma_db_collection(username: str, filename: str = 'lessons/class_11_physics.json'):
    collection_name = f"{username}_lessons"
    logging.info(f"Building ChromaDB collection for {filename} with name '{collection_name}'")
    lessons = load_json_data(filename)
    documents = [
        f"{lesson.get('unit', '')} {lesson.get('topic_title', '')} {lesson.get('description', '')} {lesson.get('elaboration', '')}"
        for lesson in lessons
    ]
    logging.info(f"Encoding {len(documents)} documents for embeddings")
    try:
        embeddings = embedding_model.encode(documents, show_progress_bar=True).tolist()
        logging.info(f"Encoded {len(embeddings)} embeddings FROM LOCAL DB!")
    except Exception as e:
        logging.error(f"Error encoding documents for embeddings in build_chroma_db_collection: {e}")
        return

    ids = [str(lesson.get('topic_id', i)) for i, lesson in enumerate(lessons)]
    metadatas = []
    for lesson in lessons:
        metadata = {
            "subject": lesson.get("subject", ""),
            "grade": lesson.get("grade", 0),
            "unit": lesson.get("unit", ""),
            "topic_id": lesson.get("topic_id", ""),
            "topic_title": lesson.get("topic_title", ""),
            "keywords": lesson.get("keywords", []),
            "references": lesson.get("references", ""),
            "hours": lesson.get("hours", 0),
            "type": lesson.get("type", ""),
            'description': lesson.get('description', ''),
            'elaboration': lesson.get('elaboration', '')
        }
        # Clean any None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        metadatas.append(metadata)

    client = chromadb.PersistentClient(path='./tmp/local VDB/chromadb')
    logging.info("Connecting to ChromaDB")
    collection = client.get_or_create_collection(name=collection_name)
    try:
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=[sanitize_metadata(metadata) for metadata in metadatas]
        )
        logging.info(f"ChromaDB collection '{collection_name}' built successfully")
    except Exception as e:
        logging.error(f"Error adding documents to ChromaDB collection '{collection_name}': {e}")


def save_scraped_data_to_vdb(
        username: str,
        scraped_file: str = "raw_data.json",
        vdb_path: str = "/tmp/local VDB/chromadb"
):
    collection_name = f"{username}_scraped_data"
    logging.info(f"Loading scraped data from {scraped_file}")
    scrapped_data = load_json_data(scraped_file)
    scrapped_documents = [f"{item.get('main_findings')} {item.get('keywords')} {item.get('headings')}" for item in
                          scrapped_data]

    try:
        embeddings = embedding_model.encode(scrapped_documents, show_progress_bar=True).tolist()
        logging.info(f"Encoded {len(embeddings)} embeddings OF SCRAPPED DATA!")
    except Exception as e:
        logging.error(f"Error encoding scraped documents for embeddings in save_scraped_data_to_vdb: {e}")
        return

    scrapped_meta = [
        {
            "headings": item.get("headings", []),
            "main_findings": item.get("main_findings", []),
            "keywords": item.get("keywords", []),
        }
        for item in scrapped_data
    ]

    client = chromadb.PersistentClient(path=vdb_path)
    collection = client.get_or_create_collection(collection_name)

    try:
        collection.add(
            ids=[str(i) for i in range(1, len(scrapped_data) + 1)],
            embeddings=embeddings,
            documents=scrapped_documents,
            metadatas=[sanitize_metadata(_) for _ in scrapped_meta]
        )
        logging.info(f"ChromaDB collection '{collection_name}' built successfully")
    except Exception as e:
        logging.error(f"Error adding scraped documents to ChromaDB collection '{collection_name}': {e}")


def get_user_courses(username: str, vdb_path: str = "/tmp/local VDB/chromadb"):
    lessons_collection_name = f"{username}_lessons"
    client = chromadb.PersistentClient(path=vdb_path)
    try:
        lessons_col = client.get_collection(lessons_collection_name)
        return lessons_col.get()
    except Exception as e:
        logging.error(f"Error getting user courses: {e}")
        return None
