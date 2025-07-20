from langchain_core.prompts import PromptTemplate
import json
from models.llm_models import MODEL


class UserSummaryTemplate(PromptTemplate):
    def __init__(self):
        super().__init__(
            template=(
                """Your role is {action}. You are given this user data:
{existing_data}

For each field, create a natural language description of the data.

Return the result as a JSON object with the same keys,
but values must be friendly sentences that summarize the meaning.

Example output format:
{{
  "username": "User's name is dyane_master",
  "age": "User is 22 years old",
  "grade": "User is halfway through Grade 12",
  ...
}}

Don't add any explanation outside the JSON.
"""))

    def format_prompt(self, action: str, existing_data: dict):
        return self.format(
            action=action,
            existing_data=existing_data
        )


class LearningResourceTemplate(PromptTemplate):
    def __init__(self):
        super().__init__(
            template=(
"""Your role is {action}. Based on this learning resource:
{existing_data}

Return it as a JSON string. Don't include any additional text or explanations.
Return the summarised user_info."""
            ),
            input_variables=["action", "existing_data"]
        )

    def format_prompt(self, action: str, existing_data: dict):
        return self.format(
            action=action,
            existing_data=existing_data
        )


prompt_user = UserSummaryTemplate()
prompt_resource = LearningResourceTemplate()

user_summary = prompt_user | MODEL
learning_resource = prompt_resource | MODEL