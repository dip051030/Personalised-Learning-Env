import json
import logging
from nodes import graph_run
from utils.utils import save_learning_state_to_json


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
        "user_info": "A motivated grade 12 student interested in physics."
    },
    "current_resource": {
        "subject": "physics",
        "grade": 12,
        "unit": "Mechanics",
        "topic_id": "6.4",
        "topic": "Centripetal force",
        "description": "Define and calculate centripetal force for systems like satellites in orbit or vehicles on curved roads.",
        "elaboration": "Define and calculate centripetal force for systems like satellites in orbit or vehicles on curved roads.",
        "keywords": ["centripetal force", "circular motion", "satellite orbits"],
        "hours": 9,
        "references": "Page 46"
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
    # Save the learning state to a JSON file
    save_learning_state_to_json(output, "learning_state.json")

if __name__ == "__main__":
    main()
