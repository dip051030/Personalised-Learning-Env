from langgraph.graph import StateGraph

from schemas import LearningState


def user_info_node(state: LearningState) -> LearningState:
    """
    Process user data and return a summary.
    """
    if state.user.model_dump():
        response = chain.invoke({"user_data": state.user.model_dump_json(indent=2)})