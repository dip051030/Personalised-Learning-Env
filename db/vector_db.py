import chromadb
from sentence_transformers import SentenceTransformer
from db.loader import load_lesson_data
import logging

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


def build_chroma_db_collection(filename: str, collection_name: str = 'lessons'):
    """
    Build a ChromaDB collection from lesson data and return the collection and embedding model.

    Args:
        filename (str): The lesson data filename.
        collection_name (str): The name for the ChromaDB collection.

    Returns:
        tuple: (collection, embedding model)
    """
    logging.info(f"Building ChromaDB collection for {filename} with name '{collection_name}'")
    lessons = load_lesson_data(filename)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    documents = [
        f"{lesson.get('unit', '')} {lesson.get('topic_title', '')} {lesson.get('description', '')} {lesson.get('elaboration', '')}"
        for lesson in lessons
    ]
    logging.info(f"Encoding {len(documents)} documents for embeddings")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()
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

    client = chromadb.PersistentClient(path = './local VDB/chromadb')
    logging.info("Connecting to ChromaDB")
    collection = client.create_collection(name=collection_name)
    logging.info(f"Adding documents and embeddings to ChromaDB collection '{collection_name}'")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas= [sanitize_metadata(metadata) for metadata in metadatas]
    )
    logging.info(f"ChromaDB collection '{collection_name}' built successfully")
    return collection, model


def save_scraped_data_to_vdb(
    scraped_file: str = "./data/raw_data.json",
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
    with open(scraped_file, "r", encoding="utf-8") as f:
        scraped_data = json.load(f)


    valid_items = [item for item in scraped_data if item.get("content")]

    if not valid_items:
        logging.warning("No valid scraped items with content found.")
        return

    texts = [item["content"] for item in valid_items]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=vdb_path)
    collection = client.get_or_create_collection(collection_name)

    for idx, (item, embedding) in enumerate(zip(valid_items, embeddings)):
        collection.add(
            ids=[f"scraped_{idx}"],
            embeddings=[embedding],
            metadatas=[item],
            documents=[item["content"]]
        )
    logging.info(f"Added {len(valid_items)} scraped items to ChromaDB collection '{collection_name}'")