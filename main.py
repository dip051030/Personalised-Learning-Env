from prompts.prompts import chain
import json
from nodes import graph_run

user_data =  {
    'username': 'dyane_master',
    'age': 22,
    'grade': 12.5,
    'id': 1,
    'is_active': True,
    'topic': 'math',
    'subtopic': 'algebra',
}


output = graph_run(user_data)
print(output)