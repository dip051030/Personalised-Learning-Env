from prompts.prompts import chain
import json

user_data =  {
    'username': 'dyane_master',
    'age': 22,
    'grade': 12.5,
    'id': 1,
    'is_active': True,
    'topic': 'math',
    'subtopic': 'algebra',
}

result = chain.invoke({"user_data": json.dumps(user_data, indent=2)})

print(result.content)