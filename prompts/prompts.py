from langchain_core.prompts import PromptTemplate
import json

from pydantic import BaseModel

from models.llm_models import MODEL
from schemas import UserInfo


# ---------------------------------------------------------------------------------
# 🧠 UserSummaryTemplate: Prompt to turn raw user data into natural-language JSON
# ---------------------------------------------------------------------------------

class UserSummaryTemplate(PromptTemplate):
    """
    Template for summarizing user data. The LLM is instructed to return natural
    language descriptions for **each key** in the user object, including fields
    like `username`, `age`, `grade`, `is_active`, `topic`, and `subtopic`.

    It returns a JSON object with the same keys, but values are reworded explanations.
    """

    def __init__(self):
        super().__init__(
            template=(
                """Your role is {action}. You are given structured user data:
{existing_data}

Your task:
- For **every key-value pair**, write a meaningful, human-readable summary.
- Maintain the same keys in the output.
- Summarize any field, including: `username`, `age`, `grade`, `is_active` or any others present.

Return the result as a JSON formatted object only. Do **not** add explanations outside of the JSON.

Example output:
{{
  "username": "User's name is dyane_master",
  "age": "User is 22 years old",
  "grade": "User is halfway through Grade 12",
  "is_active": "User is currently active",
  "user_info": "The user named dyane_master is 22 years old and is currently active. They are halfway through Grade 12 and currently focused on academic growth. No further user details are available at this time."
}}
"""
            ),
            input_variables=["action", "existing_data"]
        )

    def format_prompt(self, action: str, existing_data: dict) -> str:
        """
        Convert the input dict to a pretty JSON string for better formatting and inject it into the prompt.
        """
        return self.format(
            action=action,
            existing_data=json.dumps(existing_data, indent=2)
        )

# -----------------------------------------------------------------------------------------
# 📚 LearningResourceTemplate: Summarizes learning content + links it to user interest
# -----------------------------------------------------------------------------------------

class LearningResourceTemplate(PromptTemplate):
    """
    Template for summarizing a learning resource. The LLM is expected to:
    - Convert each key into a natural language explanation
    - Relate the learning resource back to the user's interests or context
    """

    def __init__(self):
        super().__init__(
            template=(
                """Your role is {action}. Based on this structured learning resource data:
{existing_data}
{current_resources_data}

Your task:
- For every field present (e.g., `topic`, `subtopic`), summarize it in natural language.
- Relate the content back to the user's needs or learning goals if applicable.
- Maintain the same keys as input.

Return the result as a JSON object. Do not include explanations outside the JSON.
"""
            ),
            input_variables=["action", "existing_data", 'current_resources_data']
        )

    def format_prompt(self, action: str, existing_data: dict) -> str:
        """
        Convert input data to formatted string and inject into prompt.
        """
        return self.format(
            action=action,
            existing_data=json.dumps(existing_data, indent=2),
            current_resources_data=json.dumps(existing_data.get('current_resources_data', {}), indent=2)
        )

# -----------------------------------------------------------------------------------------
# 🔗 Build chain objects by piping prompt → model
# These can be used directly in LangGraph or LangChain agents
# -----------------------------------------------------------------------------------------

from langchain_core.prompts import PromptTemplate
import json


prompt_user = UserSummaryTemplate()
prompt_resource = LearningResourceTemplate()

user_summary = (prompt_user | MODEL)
learning_resource = (prompt_resource | MODEL)
