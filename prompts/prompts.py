from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from models.llm_models import MODEL

# 1. Create a proper prompt template
class UserSummaryTemplate(PromptTemplate):
    def __init__(self):
        super().__init__(
            template="You summarize user behavior based on this data:\n{user_data}\n\nWrite a 3-sentence summary about the user:",
            input_variables=["user_data"]
        )

    def format_prompt(self, user_data: str) -> str:
        return self.format(user_data=user_data)



prompt =  UserSummaryTemplate()

# 2. Make the chain
chain = prompt | MODEL