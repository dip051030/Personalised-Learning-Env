"""main.py

This script serves as the entry point for the Personalised Learning System. It initializes the learning graph,
processes user data, generates educational content, and saves the learning state and generated content.
It orchestrates the flow of various nodes defined in `nodes.py` to create a dynamic learning experience.
"""
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from nodes import graph_run

user_data = {
    "user": {
        "username": "student_01",
        "age": 17,
        "grade": 12,
        "id": 101,
        "is_active": True,
        "user_info": "A motivated grade 11 student interested in physics."
    },
    "current_resource": {
        "subject": "physics",
        "grade": 11,
        "unit": "",
        "topic_id": "",
        "topic": "Magnetism",
        "description": "",
        "elaboration": "",
        "keywords": [],
        "hours": 7,
        "references": ""
    },
    "progress": [],
    "next_action": {"next_node": "lesson_blog"},
    "history": [],
    "enriched_resource": None,
    "topic_data": None,
    "related_examples": None,
    "content_type": "lesson",
    "content": None,
    "feedback": None
}


async def main():
    """
    Asynchronous entry point for running the learning graph.

    This function orchestrates the entire learning content generation process:
    1. Invokes the `graph_run` function with predefined `user_data` to generate learning content.
    2. Validates and converts the output to a `LearningState` object.
    3. Saves the final `LearningState` to `learning_state.json`.
    4. Extracts and saves the generated educational content to `generated_content.md` if available.

    Logs the progress and any errors encountered during the process.
    """
    output = await graph_run(user_data)
    logging.info(f"Graph has given an output! {output}")
    logging.info(f"Output content: {output.get('content').content if output.get('content') else 'No content found'}")
    # If output is not a LearningState, convert it
    from schemas import LearningState
    if not isinstance(output, LearningState):
        try:
            output = LearningState.model_validate(output)
        except Exception as e:
            logging.error(f"Failed to convert output to LearningState: {e}")
            print("Error: Output is not a valid LearningState object.")
            return
    # Save the learning state to a JSON file
    from utils.utils import save_learning_state_to_json, save_generated_content
    save_learning_state_to_json(output, "learning_state.json")
    # Save the generated content to a separate file if available
    if output.content and getattr(output.content, 'content', None):
        save_generated_content(output.content.content, "generated_content.md")
        print("Generated content saved to generated_content.md")
    else:
        print("No generated content to save.")


if __name__ == "__main__":
    asyncio.run(main())
