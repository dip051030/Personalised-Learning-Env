import json
from nodes import graph_run


user_data = {
    "user": {
        "username": "dyane_master",
        "age": 22,
        "grade": 12.5,
        "id": 1,
        "is_active": True
    },
    "current_resource": {
        "topic": "math",
        "subtopic": "algebra"
    },
    "progress": [
        {
            "id": 101,
            "resource_topic": {
                "topic": "math",
                "subtopic": "algebra"
            },
            "completed": False
        }
    ],
    "next_action": "select_resource",
    "history": []
}

output = graph_run(user_data)
print(output)