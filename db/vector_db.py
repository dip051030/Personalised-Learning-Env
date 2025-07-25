import chromadb
from sentence_transformers import SentenceTransformer
from db.loader import load_lesson_data

def build_chroma_db_collection(filename: str, collection_name: str = 'lessons'):
    lessons = load_lesson_data(filename)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    documents = [
        f"{lesson.get('unit', '')} {lesson.get('topic_title', '')} {lesson.get('description', '')}"
        for lesson in lessons
    ]
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
            "type": lesson.get("type")
        }
        
        for lesson in lessons
    ]

    client = chromadb.Client()
    collection = client.create_collection(name=collection_name)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    return collection, model
