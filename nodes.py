from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langsmith import expect
from pydantic_core import ValidationError
from setuptools.namespaces import flatten
from sympy.stats import Expectation
import logging

from logis.logical_functions import lesson_decision_node, blog_decision_node, parse_chromadb_metadata, \
    retrieve_and_search
from prompts.prompts import user_summary, enriched_content, \
    content_improviser, CONTENT_IMPROVISE_SYSTEM_PROMPT, route_selector, blog_generation, content_generation, \
    CONTENT_FEEDBACK_SYSTEM_PROMPT
from schemas import LearningState, ContentResponse, EnrichedLearningResource
import  json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def user_info_node(state: LearningState) -> LearningState:
    logging.info("Entering user_info_node")
    if state.user is not None:
        try:
            response = user_summary.invoke({
                "action": "summarise_user",
                "existing_data": state.user.model_dump()
            })
            print(response)
            user_data = response.content if hasattr(response, 'content') else response
            print('hello')
            # print(response)
            state.user = state.user.model_validate(user_data if isinstance(user_data, dict) else user_data.model_dump())

            logging.info(f"User info processed: {state.user}")
        except Exception as e:
            logging.error(f"Error processing user data: {e}")
    return state


def enrich_content(state: LearningState) -> LearningState:
    logging.info("Entering enrich_content node")
    if state.current_resource is not None:
        try:
            retrieved_content = retrieve_and_search(state=state).get('metadatas', [])
            # print(state.current_resource)
            retrieved_content = list(flatten(retrieved_content))[0]
            print('RETRIEVED CONTENT', retrieved_content)
            response= enriched_content.invoke({
                "action": "content_enrichment",
                "current_resources_data": parse_chromadb_metadata(retrieved_content).model_dump()
            })
            # print('RESPONSE FROM THE MODEL:', state.enriched_resource.model_validate(response))
            resource_data = response.content if hasattr(response, "content") else response
            state.enriched_resource = EnrichedLearningResource.model_validate(resource_data)
            logging.info(f"Learning resource processed: {state.enriched_resource}")
        except Exception as e:
            # print('METADATA:     ', parse_chromadb_metadata(retrieved_content).model_dump())
            logging.error(f"Error processing learning resource data: {e}")

    return state



def route_selector_node(state: LearningState) -> LearningState:
    logging.info("Entering route_selector_node")
    if state.user is not None and state.current_resource is not None:
        try:
            logging.info(f"Selecting the route for resource: {state.current_resource}")
            response = route_selector.invoke({
                'current_resources' : state.enriched_resource.model_dump()
            })
            # state.content_type = ContentResponse.LESSON if decision_node(state) == "lesson_selection" else ContentResponse.BLOG
            state.next_action = response.content if hasattr(response, "content") else response
            logging.info(f"Route selection response: {state.next_action}")

        except Exception as e:
            logging.error(f"Error selecting route: {e}")
    return state


def generate_lesson_content(state: LearningState) -> LearningState:
    logging.info("Entering generate_lesson_content node")
    if state.user is not None and state.enriched_resource is not None:
        try:
            logical_response = lesson_decision_node(state=state)
            logging.info(f"Logical response for lesson generation: {logical_response}")
            response = content_generation.invoke({
                "action": "generate_lesson",
                "user_data": state.user.model_dump(),
                "resource_data": state.enriched_resource.model_dump(),
                "style": logical_response
            })
            resource_data = response.content if hasattr(response, "content") else response
            print(f'Generated Content: {resource_data}')
            # state.content.content = resource_data
            logging.info(f"Lesson content has been generated!")
        except Expectation as e:
            logging.error(f"Error generating lesson content: {e}")
    return state


def generate_blog_content(state: LearningState) -> LearningState:
    logging.info("Entering generate_blog_content node")
    if state.user is not None and state.enriched_resource is not None:

        try:
            logical_response = blog_decision_node(state=state)
            logging.info(f"Logical response for blog generation: {logical_response}")
            response = blog_generation.invoke({
                "action": "generate_lesson",
                "user_data": state.user.model_dump(),
                "resource_data": state.enriched_resource.model_dump(),
                "style": logical_response
            })

            # state.content.content = ContentResponse(content=response.content if hasattr(response, "content") else response)
            logging.info(f"Blog content has been generated!")
        except Expectation as e:
            logging.error(f"Error generating blog content: {e}")
    return state


def content_improviser_node(state: LearningState) -> LearningState:
    logging.info("Entering content_improviser_node")
    if state.content is not None:
        try:
            messages = [
                CONTENT_IMPROVISE_SYSTEM_PROMPT,
                HumanMessage(content=f"""
Unpolished Learning Resource:
{state.content.model_dump()}
""")
            ]

            response = content_improviser(messages)
            generated_markdown = response.content if hasattr(response, "content") else str(response)
            state.content.content = ContentResponse(content=generated_markdown)
            logging.info(f"Improvised content has been generated!")
        except Exception as e:
            logging.error(f"Error improvising content: {e}")

    return state


def collect_feedback_node(state:LearningState) -> LearningState:
    logging.info("Entering collect_feedback_node")
    if state.content is not None:
        try:
            messages = [
                CONTENT_FEEDBACK_SYSTEM_PROMPT,
                HumanMessage(content=f"""
Feedback:
{state.content.content}

""")
            ]
            response = messages
            logging.info(f"Collecting feedback for content: {state.content.content}")
            logging.info(f"Response: {response}")
            # Assume feedback is collected and processed
        except Exception as e:
            logging.error(f"Error collecting feedback: {e}")
    return state

builder = StateGraph(LearningState)
builder.add_node("user_info", user_info_node)
builder.add_node("learning_resource", enrich_content)
builder.add_node("route_selector", route_selector_node)
builder.add_node("content_generation", generate_lesson_content)
builder.add_node("blog_generation", generate_blog_content)
builder.add_node("content_improviser", content_improviser_node)

builder.set_entry_point("user_info")
builder.add_edge("user_info", "learning_resource")
builder.add_edge("learning_resource", "route_selector")
builder.add_conditional_edges(
    "route_selector",
    lambda state: "blog_generation" if state.next_action == "blog" else "content_generation",
    {
        "blog_generation": "blog_generation",
        "content_generation": "content_generation"
    }
)
builder.add_edge("content_generation", "content_improviser")
builder.add_edge("blog_generation", "content_improviser")
builder.add_edge("content_improviser", END)

graph = builder.compile()

def graph_run(user_data: dict):
    return graph.invoke(LearningState.model_validate(user_data))
