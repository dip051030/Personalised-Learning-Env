from schemas import LearningState


def decision_node(state: LearningState) -> str:
    """
    Placeholder for a decision-making node.
    This function can be expanded to include logic for making decisions based on the state.
    """
    # Implement decision logic here
    subtopic = state.current_resource.subtopic
    grade = int(state.user.grade)

    blog_keywords = []
    lesson_keywords = []

    if any(kw in subtopic for kw in blog_keywords) and grade > 10:
        return "blog"
    elif any(kw in subtopic for kw in lesson_keywords):
        return "lesson"
    elif grade <= 10:
        return "lesson"
    else:
        return "lesson"
