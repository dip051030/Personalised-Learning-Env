from more_itertools import flatten
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from logis.logical_functions import lesson_decision_node, blog_decision_node, parse_chromadb_metadata, \
    retrieve_and_search, update_content_count
from prompts.prompts import user_summary, enriched_content, \
    content_improviser, CONTENT_IMPROVISE_SYSTEM_PROMPT, route_selector, blog_generation, content_generation, \
    CONTENT_FEEDBACK_SYSTEM_PROMPT, prompt_content_improviser, prompt_feedback, content_feedback, gap_finder
from schemas import LearningState, ContentResponse, EnrichedLearningResource, FeedBack, RouteSelector
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def user_info_node(state: LearningState) -> LearningState:
    """
    Node to process and summarize user information using the user_summary prompt.
    Updates the state with validated user info.
    """
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
            state.user = state.user.model_validate(user_data if isinstance(user_data, dict) else user_data.model_dump())
            logging.info(f"User info processed: {state.user}")
        except Exception as e:
            logging.error(f"Error processing user data: {e}")
    return state


def enrich_content(state: LearningState) -> LearningState:
    """
    Node to enrich the current learning resource using LLM enrichment.
    Updates the state with an enriched resource.
    """
    logging.info("Entering enrich_content node")
    if state.current_resource is not None:
        try:
            retrieved = retrieve_and_search(state=state)
            if not retrieved:
                logging.error("No content retrieved from vector DB.")
                return state
            retrieved_content = retrieved.get('metadatas', [])
            if not retrieved_content:
                logging.error("No metadatas found in retrieved content.")
                return state
            retrieved_content = list(flatten(retrieved_content))[0]
            print('RETRIEVED CONTENT', retrieved_content)
            response = enriched_content.invoke({
                "action": "content_enrichment",
                "current_resources_data": parse_chromadb_metadata(retrieved_content).model_dump()
            })
            resource_data = response.content if hasattr(response, "content") else response
            state.enriched_resource = EnrichedLearningResource.model_validate(resource_data)
            logging.info(f"Learning resource processed: {state.enriched_resource}")
        except Exception as e:
            logging.error(f"Error processing learning resource data: {e}")
    return state


def route_selector_node(state: LearningState) -> LearningState:
    """
    Node to select the next route (lesson or blog) based on the enriched resource.
    Updates the state with the next action.
    """
    logging.info("Entering route_selector_node")
    if state.user is not None and state.current_resource is not None:
        try:
            logging.info(f"Selecting the route for resource: {state.current_resource}")
            response = route_selector.invoke({
                'current_resources': state.enriched_resource.model_dump()
            })
            # Set next_action as a RouteSelector model
            next_action_str = response.content if hasattr(response, "content") else response
            state.next_action = RouteSelector(next_node=next_action_str)
            logging.info(f"Route selection response: {state.next_action}")
        except Exception as e:
            logging.error(f"Error selecting route: {e}")
    return state


def generate_lesson_content(state: LearningState) -> LearningState:
    """
    Node to generate lesson content using the content_generation prompt.
    """
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
            # print(f'Generated Content: {resource_data}')
            state.content = ContentResponse(content = resource_data)
            logging.info(f"Lesson content has been generated!")
            print(f'Generated Content: {state.content}')
        except Exception as e:
            logging.error(f"Error generating lesson content: {e}")
    return state


def generate_blog_content(state: LearningState) -> LearningState:
    """
    Node to generate blog content using the blog_generation prompt.
    """
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
            resource_data = response.content if hasattr(response, "content") else response
            state.content = ContentResponse(content=resource_data)
            logging.info(f"Blog content has been generated!")
        except Exception as e:
            logging.error(f"Error generating blog content: {e}")
    return state


def content_improviser_node(state: LearningState) -> LearningState:
    """
    Node to improve generated content using the content improver LLM.
    Uses the latest feedback (including gaps) to improve the content.
    Updates the state with improved content.
    """
    logging.info("Entering content_improviser_node")
    if state.content is not None and state.feedback is not None:
        try:
            messages = [
                prompt_content_improviser,
                HumanMessage(content=f"""
Unpolished Learning Resource:
{state.content.model_dump()}

Please improve the content by making it more engaging, informative, and suitable for the target audience.

Feedback (including gaps):
{state.feedback.model_dump()}

""")
            ]
            response = content_improviser.invoke(messages)
            improved_content = response.content if hasattr(response, "content") else str(response)
            # Update state.content with the newly generated improvised content
            state.content = ContentResponse(content=improved_content)
            logging.info(f"Improvised content has been generated and updated in state.content!")
        except Exception as e:
            logging.error(f"Error improvising content: {e}")
    return state


def collect_feedback_node(state:LearningState) -> LearningState:
    """
    Node to collect feedback on generated content.
    Always uses the latest content and updates state.feedback with new feedback.
    """
    logging.info("Entering collect_feedback_node")
    feedback_data = None
    if state.content is not None:
        try:
            logging.info("Collecting feedback for content")
            messages = [
                prompt_feedback,
                HumanMessage(content=f"""
Unpolished Learning Resource:
{state.content.content}
""")
            ]
            response = content_feedback.invoke(messages)
            logging.info("Feedback has been collected!")
            feedback_data = response.content if hasattr(response, "content") else response
            feedback_data = json.loads(feedback_data) if isinstance(feedback_data, str) else feedback_data
            state.feedback = FeedBack.model_validate(feedback_data)
            logging.info(f"Feedback processed and updated: {state.feedback}")
            # Log rating and gaps for debugging
            logging.info(f"Feedback rating: {state.feedback.rating}, gaps: {state.feedback.gaps}")
        except Exception as e:
            logging.error(f"Error collecting feedback: {e}")
    return state


def find_content_gap_node(state: LearningState) -> LearningState:
    """
    Node to find content gaps based on user feedback.
    Updates the feedback in state with new gaps for the next improvise node.
    """
    logging.info("Entering find_content_gap_node")
    if state.feedback is not None and state.content is not None:
        logging.info(f"Finding content gaps based on feedback: {state.feedback}")
        data = gap_finder.invoke({
            'content': state.content.content if hasattr(state.content, 'content') else str(state.content),
            'feedback': state.feedback.model_dump(),
        })
        response = data.content if hasattr(data, "content") else data
        print(f'Gaps : {response}')
        # Update feedback with new gaps for the next improvise node
        updated_feedback = FeedBack.model_validate(json.loads(response) if isinstance(response, str) else response)
        state.feedback = updated_feedback
        logging.info(f"Feedback received and updated: {state.feedback}")
        # Log rating and gaps for debugging
        logging.info(f"GapFinder rating: {state.feedback.rating}, gaps: {state.feedback.gaps}")
    return state

def update_state(state: LearningState) -> LearningState:
    try:
        if not hasattr(state, "count"):
            state.count = 0
        response = update_content_count(state)
        if response == 'Update required':
            state.count += 1
            logging.info(f"State updated: {state.count}")
        else:
            logging.info(f"No update required, current count: {state.count}")
    except Exception as e:
        logging.error(f"Error updating state: {e}")
    return state


def save_learning_state_to_json(state, file_path):
    """
    Save the details of the LearningState object to a JSON file.
    If the file does not exist, it will be created.
    Args:
        state: LearningState object (should have .model_dump() or .dict() method)
        file_path: Path to the JSON file
    """
    try:
        # Use model_dump if available (Pydantic v2), else fallback to dict
        if hasattr(state, 'model_dump'):
            state_data = state.model_dump()
        elif hasattr(state, 'dict'):
            state_data = state.dict()
        else:
            raise ValueError("State object does not support serialization.")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=4, ensure_ascii=False)
        logging.info(f"LearningState saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save LearningState to {file_path}: {e}")


builder = StateGraph(LearningState)
builder.add_node("user_info", user_info_node)
builder.add_node("learning_resource", enrich_content)
builder.add_node("route_selector", route_selector_node)
builder.add_node("content_generation", generate_lesson_content)
builder.add_node("blog_generation", generate_blog_content)
builder.add_node("content_improviser", content_improviser_node)
builder.add_node("collect_feedback", collect_feedback_node)
builder.add_node("find_content_gap", find_content_gap_node)
builder.add_node("update_state", update_state)

builder.set_entry_point("user_info")
builder.add_edge("user_info", "learning_resource")
builder.add_edge("learning_resource", "route_selector")
builder.add_conditional_edges(
    "route_selector",
    lambda state: (
        state.next_action.next_node
        if hasattr(state.next_action, "next_node") and state.next_action.next_node in ["blog_generation", "content_generation"]
        else "content_generation"  # Default branch if next_action is missing or invalid
    ),
    {
        "blog_generation": "blog_generation",
        "content_generation": "content_generation"
    }
)
builder.add_edge("content_generation", "content_improviser")
builder.add_edge("blog_generation", "content_improviser")
builder.add_edge("content_improviser", 'collect_feedback')
builder.add_edge("collect_feedback", "find_content_gap")
builder.add_edge("find_content_gap", "update_state")
builder.add_conditional_edges(
    "update_state",
    lambda state: "content_improviser" if getattr(state, "count", 0) < 4 else "END",
    {
        "content_improviser": "content_improviser",
        "END": END
    }
)

graph = builder.compile()

def graph_run(user_data: dict):
    return graph.invoke(LearningState.model_validate(user_data))
