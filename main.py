import json
import logging
from nodes import graph_run


logging.basicConfig(level=logging.INFO)

user_data = {
    "user": {
        "username": "dyane_master",
        "age": 22,
        "grade": 12.5,
        "id": 1,
        "is_active": True
    },
    "current_resource": {
        "subject": "math",
        "topic": "algebra"
    },
    "progress": [
        {
            "id": 101,
            "resource_topic": {
                "subject": "math",
                "topic": "algebra"
            },
            "completed": False
        }
    ],
    "next_action": "select_resource",
    "history": []
}

output = graph_run(user_data)
logging.info(f"Graph output: {output}")