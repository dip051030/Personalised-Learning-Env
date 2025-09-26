"""nodes.py

This module defines the individual nodes and the overall graph structure for the
personalized learning system. Each node represents a step in the content generation
and refinement pipeline, processing a `LearningState` object and updating it.
"""
import json
import logging
import os

import pydantic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from more_itertools import flatten

from logis.logical_functions import lesson_decision_node, blog_decision_node, parse_chromadb_metadata, \
    update_content_count, search_both_collections
from prompts.prompts import user_summary, enriched_content, \
    content_improviser, route_selector, blog_generation, content_generation, \
    prompt_content_improviser, prompt_feedback, content_feedback, gap_finder, \
    content_seo_optimization, prompt_post_validation, post_validation, prompt_seo_optimization
from schemas import LearningState, ContentResponse, EnrichedLearningResource, FeedBack, RouteSelector, \
    PostValidationResult
from scrapper.crawl4ai_scrapping import crawl_and_extract_json
from scrapper.save_to_local import serper_api_results_parser, save_to_local
from utils.utils import read_from_local


def user_info_node(state: LearningState) -> LearningState:
    """
    Processes and summarizes user information.

    This node takes the current `LearningState`, extracts user data, and uses an
    LLM (via `user_summary` prompt) to generate a summarized version of the user's profile.
    The summarized user information is then validated and updated back into the `state.user` attribute.

    Args:
        state (LearningState): The current state of the learning process, containing user information.

    Returns:
        LearningState: The updated state with processed and summarized user information.
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
        except pydantic.ValidationError as e:
            logging.error(f"Pydantic validation error in user_info_node: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in user_info_node: {e}")
    return state


# def enrich_content(state: LearningState) -> LearningState:
#     """
#     Node to enrich the current learning resource using LLM enrichment.
#     Updates the state with an enriched resource.
#     """
#     logging.info("Entering enrich_content node")
#     if state.current_resource is not None:
#         try:
#             retrieved = retrieve_and_search(state=state)
#             if not retrieved:
#                 logging.error("No content retrieved from vector DB.")
#                 return state
#             retrieved_content = retrieved.get('metadatas', [])
#             if not retrieved_content:
#                 logging.error("No metadatas found in retrieved content.")
#                 return state
#             retrieved_content = list(flatten(retrieved_content))[0]
#             print('RETRIEVED CONTENT', retrieved_content)
#             response = enriched_content.invoke({
#                 "action": "content_enrichment",
#                 "current_resources_data": parse_chromadb_metadata(retrieved_content).model_dump()
#             })
#             resource_data = response.content if hasattr(response, "content") else response
#             state.enriched_resource = EnrichedLearningResource.model_validate(resource_data)
#             logging.info(f"Learning resource processed: {state.enriched_resource}")
#         except Exception as e:
#             logging.error(f"Error processing learning resource data: {e}")
#     return state

async def crawler_node(state: LearningState) -> LearningState:
    """
    Crawls web data related to the current topic and updates the learning state.

    This node first checks if raw data already exists locally. If so, it loads the data.
    Otherwise, it uses `serper_api_results_parser` to get links, then `crawl_and_extract_json`
    to scrape content from those links. The extracted raw data is saved locally and
    then updated into the `state.topic_data` attribute.

    Args:
        state (LearningState): The current state of the learning process.

    Returns:
        LearningState: The updated state with crawled topic data.
    """
    if os.path.exists('./data/raw_data.json'):
        logging.info("Raw data already exists, loading from file.")
        try:
            with open('./data/raw_data.json', 'r', encoding='utf-8') as f:
                state.topic_data = json.load(f)
            logging.info("Raw data loaded from file and state updated successfully.")
            return state
        except FileNotFoundError:
            logging.error("raw_data.json not found, despite os.path.exists returning True. This is unexpected.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from raw_data.json: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading raw_data.json: {e}")
            # If loading fails, proceed with crawling
            pass
    links = serper_api_results_parser(state=state)
    logging.info(f"Scrapped Links: {links}")
    raw_data = None
    try:
        save_to_local(links, "./data/scrapped_data.json")
        link_list = [item.get('link') for item in links.get('organic', []) if 'link' in item]
        if not link_list:
            logging.warning("No valid links found for crawling.")
            return state
        raw_data = await crawl_and_extract_json(link_list)
        logging.info("Raw Data has been extracted!")
    except Exception as e:
        logging.error(f"Error extracting raw data: {e}")
        return state

    if raw_data:
        try:
            save_to_local(raw_data, "./data/raw_data.json")
            state.topic_data = raw_data
            logging.info("Raw data saved and state updated successfully.")
        except Exception as e:
            logging.error(f"Error saving raw data: {e}")
    else:
        logging.warning("No raw data extracted from the links.")
    return state


def enrich_content(state: LearningState) -> LearningState:
    """
    Enriches the current learning resource using retrieved data and LLM capabilities.

    This node retrieves relevant data from both local and scraped data collections,
    then invokes an LLM (via `enriched_content` prompt) to enrich the `current_resource`.
    The enriched data is validated against `EnrichedLearningResource` schema and
    updated into the `state.enriched_resource` attribute.

    Args:
        state (LearningState): The current state of the learning process.

    Returns:
        LearningState: The updated state with an enriched learning resource.
    """
    logging.info("Entering enrich_content node")
    if state.current_resource is not None:
        try:
            retrieved_data = search_both_collections(state=state)
            local_medadata = retrieved_data.get('lessons_results').get('metadatas')
            scrapped_metadata = retrieved_data.get('scraped_results').get('metadatas')
            local_medadata = list(flatten(local_medadata))[0]
            scrapped_metadata = list(flatten(scrapped_metadata))[0]
            response = enriched_content.invoke({
                "action": "content_enrichment",
                "foundation_data": parse_chromadb_metadata(local_medadata).model_dump(),
                'scrapped_data': scrapped_metadata
            })
            resource_data = response.content if hasattr(response, "content") else response
            print('ENRICHED DATA=======> ', resource_data)
            try:
                state.enriched_resource = EnrichedLearningResource.model_validate(resource_data)
                logging.info(f"Learning resource processed!")
            except Exception as validation_error:
                logging.error(f"Pydantic validation error for EnrichedLearningResource: {validation_error}")
                logging.error(f"Malformed LLM output: {resource_data}")
        except pydantic.ValidationError as e:
            logging.error(f"Pydantic validation error in enrich_content (parse_chromadb_metadata): {e}")
        except Exception as e:
            print()
            logging.error(f"An unexpected error occurred in enrich_content: {e}")
    return state


def route_selector_node(state: LearningState) -> LearningState:
    """
    Selects the next route (lesson or blog generation) based on the enriched resource.

    This node uses an LLM (via `route_selector` prompt) to decide whether the next step
    should be generating a lesson or a blog. The decision is then stored in
    `state.next_action` as a `RouteSelector` object.

    Args:
        state (LearningState): The current state of the learning process.

    Returns:
        LearningState: The updated state with the selected next action.
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
            try:
                state.next_action = RouteSelector(next_node=next_action_str)
                logging.info(f"Route selection response: {state.next_action}")
            except Exception as validation_error:
                logging.error(f"Pydantic validation error for RouteSelector: {validation_error}")
                logging.error(f"Malformed LLM output: {next_action_str}")
        except pydantic.ValidationError as e:
            logging.error(f"Pydantic validation error in route_selector_node: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in route_selector_node: {e}")
    return state


def generate_lesson_content(state: LearningState) -> LearningState:
    """
    Generates educational lesson content.

    This node orchestrates the generation of lesson content by invoking an LLM
    (via `content_generation` prompt) with user data, enriched resource data,
    a determined logical style, and relevant URLs. The generated content is then
    validated and stored in `state.content` as a `ContentResponse` object.

    Args:
        state (LearningState): The current state of the learning process.

    Returns:
        LearningState: The updated state with the generated lesson content.
    """
    logging.info("Entering generate_lesson_content node")
    if state.user is not None and state.enriched_resource is not None:
        try:
            logical_response = lesson_decision_node(state=state)
            urls = read_from_local('./data/scrapped_data.json')
            print(urls)
            logging.info(f"Logical response for lesson generation: {logical_response}")
            response = content_generation.invoke({
                "action": "generate_lesson",
                "user_data": state.user.model_dump(),
                "resource_data": state.enriched_resource.model_dump(),
                "style": logical_response,
                'urls': urls
            })
            resource_data = response.content if hasattr(response, "content") else response
            try:
                state.content = ContentResponse(content=resource_data)
                logging.info(f"Lesson content has been generated!")
                print(f'Generated Content: {state.content}')
            except Exception as validation_error:
                logging.error(f"Pydantic validation error for ContentResponse: {validation_error}")
                logging.error(f"Malformed LLM output: {resource_data}")
        except pydantic.ValidationError as e:
            logging.error(f"Pydantic validation error in generate_lesson_content: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in generate_lesson_content: {e}")
    return state


def seo_optimiser_node(state: LearningState) -> LearningState:
    """
    Optimizes the generated content for Search Engine Optimization (SEO).

    This node takes the existing content from `state.content` and uses an LLM
    (via `content_seo_optimization` prompt) to apply SEO best practices.
    The optimized content replaces the original content in `state.content`.
    Error handling is included to prevent crashes if the LLM call fails.

    Args:
        state (LearningState): The current state of the learning process.

    Returns:
        LearningState: The updated state with SEO-optimized content.
    """
    if state.content is not None:
        try:
            logging.info("Optimising the content for SEO")
            messages = [prompt_seo_optimization,
                        HumanMessage(
                            content=f"""
Generated Undiagnosed Learning Resource:
{state.content.content}
""",
                        )
                        ]
            try:
                response = content_seo_optimization.invoke(messages)
                resource_data = response.content if hasattr(response, "content") else response
                state.content = ContentResponse(content=resource_data)
                logging.info(f"Content has been optimised for SEO!")
            except Exception as e:
                logging.error(f"An error occurred during SEO optimization: {e}")
        except pydantic.ValidationError as e:
            logging.error(f"Pydantic validation error in seo_optimiser_node: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in seo_optimiser_node: {e}")

    return state


def generate_blog_content(state: LearningState) -> LearningState:
    """
    Generates educational blog content.

    This node is responsible for creating blog posts by invoking an LLM
    (via `blog_generation` prompt) with user data, enriched resource data,
    and a determined logical style. The generated content is then validated
    and stored in `state.content` as a `ContentResponse` object.

    Args:
        state (LearningState): The current state of the learning process.

    Returns:
        LearningState: The updated state with the generated blog content.
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
            try:
                state.content = ContentResponse(content=resource_data)
                logging.info(f"Blog content has been generated!")
            except Exception as validation_error:
                logging.error(f"Pydantic validation error for ContentResponse: {validation_error}")
                logging.error(f"Malformed LLM output: {resource_data}")
        except pydantic.ValidationError as e:
            logging.error(f"Pydantic validation error in generate_blog_content: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in generate_blog_content: {e}")
    return state


def content_improviser_node(state: LearningState) -> LearningState:
    """
    Improves generated content based on feedback and validation results.

    This node takes the existing content, feedback, and validation results from the `state`
    and uses an LLM (via `content_improviser` prompt) to refine the content.
    The improved content replaces the original content in `state.content`.
    Error handling is included to prevent crashes if the LLM call fails.

    Args:
        state (LearningState): The current state of the learning process.

    Returns:
        LearningState: The updated state with improved content.
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

Post_Validation Result:
{state.validation_result.model_dump()}
""")
            ]
            try:
                response = content_improviser.invoke(messages)
                improved_content = response.content if hasattr(response, "content") else str(response)
                state.content = ContentResponse(content=improved_content)
                logging.info(f"Improvised content has been generated and updated in state.content!")
            except Exception as e:
                logging.error(f"An error occurred during content improvisation: {e}")
        except pydantic.ValidationError as e:
            logging.error(f"Pydantic validation error in content_improviser_node: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in content_improviser_node: {e}")
    return state


def collect_feedback_node(state: LearningState) -> LearningState:
    """
    Collects feedback on the generated content.

    This node uses an LLM (via `content_feedback` prompt) to generate feedback
    on the current content in `state.content`. The feedback, including rating,
    comments, and identified gaps, is then validated and stored in `state.feedback`.
    Error handling is included to prevent crashes if the LLM call or JSON parsing fails.

    Args:
        state (LearningState): The current state of the learning process.

    Returns:
        LearningState: The updated state with collected feedback.
    """
    logging.info("Entering collect_feedback_node")
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
            try:
                state.feedback = FeedBack.model_validate(feedback_data)
                logging.info(f"Feedback processed and updated: {state.feedback}")
                # Log rating and gaps for debugging
                logging.info(f"Feedback rating: {state.feedback.rating}, gaps: {state.feedback.gaps}")
            except Exception as validation_error:
                logging.error(f"Pydantic validation error for FeedBack: {validation_error}")
                logging.error(f"Malformed LLM output: {feedback_data}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding error in collect_feedback_node: {e}")
        except pydantic.ValidationError as e:
            logging.error(f"Pydantic validation error in collect_feedback_node: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in collect_feedback_node: {e}")
    return state


def find_content_gap_node(state: LearningState) -> LearningState:
    """
    Identifies content gaps based on existing feedback.

    This node takes the current content and feedback from the `state` and uses an LLM
    (via `gap_finder` prompt) to identify specific content gaps or areas for improvement.
    The updated feedback, including new gaps, is then stored in `state.feedback`.
    Error handling is included for LLM call and JSON parsing failures.

    Args:
        state (LearningState): The current state of the learning process.

    Returns:
        LearningState: The updated state with identified content gaps in the feedback.
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
        try:
            # Update feedback with new gaps for the next improvise node
            updated_feedback = FeedBack.model_validate(json.loads(response) if isinstance(response, str) else response)
            state.feedback = updated_feedback
            logging.info(f"Feedback received and updated: {state.feedback}")
            # Log rating and gaps for debugging
            logging.info(
                f"GapFinder rating: {state.feedback.rating}, gaps: {state.feedback.gaps}, ai_reliability_score: {state.feedback.ai_reliability_score}")
        except Exception as validation_error:
            logging.error(f"Pydantic validation error for FeedBack: {validation_error}")
            logging.error(f"Malformed LLM output: {response}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding error in find_content_gap_node: {e}")
        except pydantic.ValidationError as e:
            logging.error(f"Pydantic validation error in find_content_gap_node: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in find_content_gap_node: {e}")
    return state


def post_validator_node(state: LearningState) -> LearningState:
    """
    Validates the generated content against predefined criteria.

    This node takes the current content from `state.content` and uses an LLM
    (via `post_validation` prompt) to assess its quality, adherence to guidelines,
    and other validation criteria. The validation result is then stored in
    `state.validation_result` as a `PostValidationResult` object.
    Error handling is included for LLM call and JSON parsing failures.

    Args:
        state (LearningState): The current state of the learning process.

    Returns:
        LearningState: The updated state with the content validation result.
    """
    logging.info("Entering post_validator_node")
    validation_result = None
    if state.content is not None:
        try:
            logging.info("Checking Validation!")
            messages = [
                prompt_post_validation,
                HumanMessage(content=f"""
Learning Resource:
{state.content.content}
""")
            ]
            response = post_validation.invoke(messages)
            logging.info("Validation has been given!")
            validation_result = response.content if hasattr(response, "content") else response
            validation_result = json.loads(validation_result) if isinstance(validation_result,
                                                                            str) else validation_result
            try:
                state.validation_result = PostValidationResult.model_validate(validation_result)
                logging.info(f"Validated and Updated: {state.feedback}")
            except Exception as validation_error:
                logging.error(f"Pydantic validation error for PostValidationResult: {validation_error}")
                logging.error(f"Malformed LLM output: {validation_result}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding error in post_validator_node: {e}")
        except pydantic.ValidationError as e:
            logging.error(f"Pydantic validation error in post_validator_node: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred in post_validator_node: {e}")
    return state


def update_state(state: LearningState) -> LearningState:
    """
    Updates an internal counter within the learning state.

    This node increments a `count` attribute in the `LearningState` if an update
    is required (as determined by `update_content_count`). This counter can be
    used to track iterations or progress within the graph.

    Args:
        state (LearningState): The current state of the learning process.

    Returns:
        LearningState: The updated state with an incremented count if an update was required.
    """
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
builder.add_node("crawler", crawler_node)
builder.add_node("content_seo_optimization", seo_optimiser_node)
builder.add_node("post_validator", post_validator_node)

builder.set_entry_point("user_info")
builder.add_edge("user_info", "crawler")
builder.add_edge("crawler", "learning_resource")
builder.add_edge("learning_resource", "route_selector")
builder.add_conditional_edges(
    "route_selector",
    lambda state: (
        state.next_action.next_node
        if hasattr(state.next_action, "next_node") and state.next_action.next_node in ["blog_generation",
                                                                                       "content_generation"]
        else "content_generation"  # Default branch if next_action is missing or invalid
    ),
    {
        "blog_generation": "blog_generation",
        "content_generation": "content_generation"
    }
)
builder.add_edge("content_generation", "content_seo_optimization")
builder.add_edge("blog_generation", "content_seo_optimization")
builder.add_edge("content_seo_optimization", "content_improviser")
builder.add_edge("content_improviser", 'collect_feedback')
builder.add_edge("collect_feedback", "find_content_gap")
builder.add_edge("find_content_gap", 'post_validator')
builder.add_edge("post_validator", "update_state")
builder.add_conditional_edges(
    "update_state",
    lambda state: "content_improviser" if getattr(state, "count", 0) < 4 else "END",
    {
        "content_improviser": "content_improviser",
        "END": END
    }
)

graph = builder.compile()


async def graph_run(user_data: dict):
    """
    Invokes the LangGraph with initial user data to start the learning process.

    This asynchronous function takes initial user data, validates it against the
    `LearningState` schema, and then invokes the compiled LangGraph (`graph`).
    The graph processes the data through its defined nodes and returns the final
    `LearningState` after execution.

    Args:
        user_data (dict): A dictionary containing the initial user information.

    Returns:
        LearningState: The final state of the learning process after the graph has run.
    """
    return await graph.ainvoke(LearningState.model_validate(user_data), config={'recursion_limit': 30})
