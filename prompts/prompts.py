from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage
from models.llm_models import MODEL

# 1. Create a proper prompt template
prompt = PromptTemplate(
    template="You summarize user behavior based on this data:\n{user_data}\n\nWrite a 3-sentance summary about the user:",
    input_variables=["user_data"]
)

# 2. Make the chain
chain = prompt | MODEL