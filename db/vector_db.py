"""
Vector database utilities for building and saving ChromaDB collections from lesson and scraped data.
"""
import logging

import chromadb
from sentence_transformers import SentenceTransformer

from db.loader import load_json_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def sanitize_metadata(metadata: dict) -> dict:
    """
    Convert list values in metadata to comma-separated strings for ChromaDB compatibility.

    Args:
        metadata (dict): Metadata dictionary.

    Returns:
        dict: Sanitized metadata dictionary.
    """
    return {k: (",".join(v) if isinstance(v, list) else v) for k, v in metadata.items()}


def clean_metadata(metadata: dict) -> dict:
    """
    Remove keys with None values from metadata.

    Args:
        metadata (dict): Metadata dictionary.

    Returns:
        dict: Cleaned metadata dictionary.
    """
    return {k: v for k, v in metadata.items() if v is not None}


def build_chroma_db_collection(filename: str = 'lessons/class_11_physics.json', collection_name: str = 'lessons'):
    """
    Build a ChromaDB collection from lesson data and return the collection and embedding model.

    Args:
        filename (str): The lesson data filename.
        collection_name (str): The name for the ChromaDB collection.

    Returns:
        tuple: (collection, embedding model)
    """
    logging.info(f"Building ChromaDB collection for {filename} with name '{collection_name}'")
    lessons = load_json_data(filename)
    model = SentenceTransformer('Shashwat13333/bge-base-en-v1.5_v4')
    documents = [
        f"{lesson.get('unit', '')} {lesson.get('topic_title', '')} {lesson.get('description', '')} {lesson.get('elaboration', '')}"
        for lesson in lessons
    ]
    logging.info(f"Encoding {len(documents)} documents for embeddings")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()
    logging.info(f"Encoded {len(embeddings)} embeddings FROM LOCAL DB!")
    ids = [str(lesson.get('topic_id', i)) for i, lesson in enumerate(lessons)]
    metadatas = [
        {
            "subject": lesson.get("subject"),
            "grade": lesson.get("grade"),
            "unit": lesson.get("unit"),
            "topic_id": lesson.get("topic_id"),
            "topic_title": lesson.get("topic_title"),
            "keywords": lesson.get("keywords"),
            "references": lesson.get("references"),
            "hours": lesson.get("hours"),
            "type": lesson.get("type"),
            'description': lesson.get('description', ''),
            'elaboration': lesson.get('elaboration', '')
        }
        for lesson in lessons
    ]

    client = chromadb.PersistentClient(path='./local VDB/chromadb')
    logging.info("Connecting to ChromaDB")
    collection = client.get_or_create_collection(name=collection_name)
    logging.info(f"Adding documents and embeddings to ChromaDB collection '{collection_name}'")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=[sanitize_metadata(metadata) for metadata in metadatas]
    )
    logging.info(f"ChromaDB collection '{collection_name}' built successfully")


def save_scraped_data_to_vdb(
        scraped_file: str = "raw_data.json",
        vdb_path: str = "./local VDB/chromadb",
        collection_name: str = "scraped_data"
):
    """
    Save scraped data from a JSON file to ChromaDB vector database.

    Args:
        scraped_file (str): Path to the scraped data JSON file.
        vdb_path (str): Path to the ChromaDB persistent directory.
        collection_name (str): Name of the ChromaDB collection.
    """
    logging.info(f"Loading scraped data from {scraped_file}")
    scrapped_data = load_json_data(scraped_file)
    scrapped_documents = [f'{item.get('main_findings')} {item.get('keywords')} {item.get('headings')}' for item in
                          scrapped_data]

    logging.info(f"Encoding {len(scrapped_documents)} documents for embeddings")
    model = SentenceTransformer("Shashwat13333/bge-base-en-v1.5_v4")
    embeddings = model.encode(scrapped_documents, show_progress_bar=True).tolist()
    logging.info(f"Encoded {len(embeddings)} embeddings OF SCRAPPED DATA!")
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

    collection.add(
        ids=[str(i) for i in range(1, len(scrapped_data) + 1)],
        embeddings=embeddings,
        documents=scrapped_documents,
        metadatas=[sanitize_metadata(_) for _ in scrapped_meta]
    )
