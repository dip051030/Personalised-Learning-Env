from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic_core import ValidationError
from sympy.stats import Expectation

from prompts.prompts import user_summary, learning_resource, user_content_generation, CONTENT_GENERATION_SYSTEM_PROMPT, content_improviser
from schemas import LearningState, ContentResponse
import  json


def user_info_node(state: LearningState) -> LearningState:
    if state.user is not None:
        try:
            response = user_summary.invoke({
                "action": "summarise_user",
                "existing_data": state.user.model_dump()
            })

            user_data = response.content if hasattr(response, 'content') else response
            state.user = state.user.model_validate(user_data if isinstance(user_data, dict) else user_data.model_dump())
            # print(state.user)
        except Exception as e:
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
            # print(state.current_resource)
        except Exception as e:
            print(f"Error processing learning resource data: {e}")

    return state


def content_generation(state: LearningState) -> LearningState:
    """
    Generate content based on the current state.
    """
    if state.user and state.current_resource:
        try:
            response= user_content_generation.invoke({
                "action": "generate_content",
                "user_data": state.user.model_dump(),
                "resource_data": state.current_resource.model_dump()
            })

            content_raw = response.content if hasattr(response, "content") else response

            # generated_content = content_data.get("generated_content", "") if isinstance(content_data,dict) else content_data
            # print('RAW LLM RESPONSE:', content_raw)
            state.generated_content = ContentResponse(content = content_raw)
            # print('GENERATED CONTENT:', state.generated_content)
            # state.history.append({
            #     "user": state.user.model_dump(),
            #     "resource": state.current_resource.model_dump(),
            #     "generated_content": generated_content
            # })
            # return content_data
        except Expectation as e:
            print(f"Error generating content: {e}")

    return state


# def conent_improviser(state: LearningState) -> LearningState:
#     """
#     Improviser for content generation.
#     """
#     if state.current_resource:
#         try:
#             response = user_content_generation.invoke({
#                 "action": "improvise_content",
#                 "resource_data": state.generated_content.model_dump()
#             })
#
#             improvised_content = response.content if hasattr(response, "content") else response
#             print('IMPROVISED CONTENT:', improvised_content)
#         except ValidationError as e:
#             print(f"Error improvising content: {e}")
#
#     return state


def content_improviser_node(state: LearningState) -> LearningState:
    if state.generated_content is not None:
        try:
            messages = [
                CONTENT_GENERATION_SYSTEM_PROMPT,
                HumanMessage(content=f"""

Unpolished Learning Resource:
{state.generated_content.model_dump()}
""")
            ]

            response = content_improviser(messages)
            generated_markdown = response.content if hasattr(response, "content") else str(response)
            print('Improvised:', generated_markdown)
            # print('GENERATED CONTENT:', state.content)
        except Exception as e:
            print(f"Error improvising content: {e}")

    return state


builder = StateGraph(LearningState)
builder.add_node("user_info", user_info_node)
builder.add_node("learning_resource", learning_resource_node)
builder.add_node("content_generation", content_generation)
builder.add_node("content_improviser", content_improviser_node)

builder.set_entry_point('user_info')
builder.add_edge("user_info", "learning_resource")
builder.add_edge("learning_resource", "content_generation")
builder.add_edge("content_generation", "content_improviser")
builder.add_edge('content_generation', END)

graph = builder.compile()

def graph_run(user_data: dict):
    return graph.invoke(LearningState.model_validate(user_data))
