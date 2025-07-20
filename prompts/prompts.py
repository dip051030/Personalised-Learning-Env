from langchain_core.prompts import PromptTemplate
import json
from models.llm_models import MODEL



class UserSummaryTemplate(PromptTemplate):
    def __init__(self):
        super().__init__(
            template=(
                "Your role is {action}. Based on this user data:\n"
                "{existing_data}\n\n"
                'Return it as a JSON string.'
            ),
            input_variables=["action", "existing_data"]
        )

    def format_prompt(self, action: str, existing_data: dict):
        return self.format(
            action=action,
            existing_data=existing_data
        )



class LearningResourceTemplate(PromptTemplate):
    def __init__(self):
        super().__init__(
            template=(
                "Your role is {action}. Based on this learning resource:\n"
                "{existing_data}\n\n"
                'Return it as a JSON string.'
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

summary = prompt_user | MODEL
chain = prompt_resource | MODEL
