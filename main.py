import json
import logging
from nodes import graph_run


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

user_data = {
    "user": {
        "username": "anonymous",
        "age": 18,
        "grade": 19,
        "id": 1,
        "is_active": True
    },
    "current_resource": {
        "subject": "physics",
        "grade": 12,
        "unit": "Mechanics",
        "topic_id": "",
        "topic": "period of pendulum",
        "description": "",
        "elaboration": "",
        "keywords": [],
        "hours": 1,
        "references": ""
    },
    "progress": [],
    "next_action": {"next_node": "lesson_selection"},  # <-- THIS IS THE FIX
    "history": []
}

output = graph_run(user_data)
logging.info(f"Graph output: {output}")