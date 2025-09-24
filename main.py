import logging
from nodes import graph_run
import logging

from nodes import graph_run

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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


def main():
    """
    Entry point for running the learning graph with sample user data.
    """
    output = graph_run(user_data)
    logging.info(f"Graph has given an output! {output}")
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
    main()
