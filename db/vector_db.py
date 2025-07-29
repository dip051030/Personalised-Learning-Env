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
    return {k: (",".join(v) if isinstance(v, list) else v) for k, v in metadata.items()}


def build_chroma_db_collection(filename: str, collection_name: str = 'lessons'):
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

    client = chromadb.Client()
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
