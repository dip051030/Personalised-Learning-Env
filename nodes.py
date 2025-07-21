from langgraph.graph import StateGraph, START, END
from mpmath import isint
from pydantic_core import ValidationError
from prompts.prompts import user_summary, learning_resource
from schemas import LearningState
from langchain_core.messages import AIMessage
import  json


def user_info_node(state: LearningState) -> LearningState:
    if state.user is not None:

        try:
            response= user_summary.invoke({
                "action": "summarise_user",
                "existing_data": state.user.model_dump()
            })

            user_data = response.content if hasattr(response, 'content') else response
            state.user = state.user.model_validate(user_data)
            # print(state.user)
        except Exception as e:
            print('State User:', state.user.model_dump())
            print('State User:', state.user.model_dump())
            print(f"Error processing user data: {e}")
    return state


def learning_resource_node(state: LearningState) -> LearningState:
    """
    Process learning resource data.
    """
    if state.current_resource is not None:
        try:
            response= learning_resource.invoke({
                "action": "summarise_resource",
                "existing_data": state.user.model_dump(),
                "current_resources_data": state.current_resource.model_dump()
            })

            resource_data = response.content if hasattr(response, "content") else response
            state.current_resource = state.current_resource.model_validate(resource_data)

        except Exception as e:
            print(f"Error processing learning resource data: {e}")
            state.current_resource = None

    return state


def content_generation(state: LearningState) -> str:
    """
    Generate content based on the current state.
    """
    if state.user and state.current_resource:
        try:
            response= chain.invoke({
                "action": "generate_content",
                "user_data": state.user.model_dump(),
                "resource_data": state.current_resource.model_dump()
            })

            content_data = json.loads(response.content if hasattr(response, "content") else response)
            state.history.append({
                "user": state.user.model_dump(),
                "resource": state.current_resource.model_dump(),
                "generated_content": content_data
            })

        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Error generating content: {e}")



builder = StateGraph(LearningState)
builder.add_node("user_info", user_info_node)
builder.add_node("learning_resource", learning_resource_node)
builder.add_node("content_generation", content_generation)

builder.set_entry_point('user_info')
builder.add_edge("user_info", "learning_resource")
builder.add_edge("learning_resource", "content_generation")
builder.add_edge('content_generation', END)

graph = builder.compile()

def graph_run(user_data: dict):
    return graph.invoke(LearningState.model_validate(user_data))
