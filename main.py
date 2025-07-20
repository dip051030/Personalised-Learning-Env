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
    "current_resource": {},  # add resource info if required
    "history": []            # or other fields as needed by your schema
}


output = graph_run(user_data)
print(output)