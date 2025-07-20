from prompts.prompts import chain
import json
from nodes import graph_run


user_data = {
    "user": {
        "username": "dyane_master",
        "age": 22,
        "grade": 12.5,
        "id": 1,
        "is_active": True,
        "topic": "math",
        "subtopic": "algebra"
    },
    "current_resource": {
        "topic": "math",
        "subtopic": "algebra"
    },
    "history": []
}

output = graph_run(user_data)
print(output)