import logging

from sympy import resultant

from schemas import LearningState
from db.vector_db import build_chroma_db_collection

def retrieve_and_search(state:LearningState) -> str:
    """
    Retrieve and search for resources based on the current state.
    This function can be expanded to include more complex search logic as needed.
    """

    try:
        if state.current_resource is not None:
            collection = build_chroma_db_collection('data/lessons/class_11_physics.json', collection_name='lessons')
            query_embedding = model.encode([state.current_resource.topic]).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=5
            )
            return results
    except Exception as e:
        logging.error(f"Error retrieving and searching resources: {e}")
        return "Error retrieving resources"


def decision_node(state: LearningState) -> str:
    """
    Placeholder for a decision-making node.
    This function can be expanded to include logic for making decisions based on the state.
    """
    # Implement decision logic here
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


def lesson_decision_node(state: LearningState) -> LearningState:
    """
    Node to handle logical functions based on the current state.
    This function can be expanded to include more complex logic as needed.
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


def blog_decision_node(state: LearningState) -> LearningState:
    """
    Node to handle logical functions based on the current state.
    This function can be expanded to include more complex logic as needed.
    """
    if "importance" in state.current_resource.topic:
        style = "motivational"
    elif state.user.grade >= 12:
        style = "application_focused"
    else:
        style = "storytelling"

    return style