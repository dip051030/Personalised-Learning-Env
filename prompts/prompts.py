from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
import json

from pydantic import BaseModel

from models.llm_models import get_gemini_model, get_groq_model
from schemas import UserInfo, LearningResource, LearningState, ContentResponse
from nodes import user_info_node, learning_resource_node, content_generation, content_improviser_node


# ---------------------------------------------------------------------------------
# 🧠 UserSummaryTemplate: Prompt to turn raw user data into natural-language JSON
# ---------------------------------------------------------------------------------

class UserSummaryTemplate(PromptTemplate):
    """
    Template for summarizing user data. The LLM is instructed to return natural
    language descriptions for **each key** in the user object, including fields
    like `username`, `age`, `grade`, `is_active`, `subject`, and `topic`.

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
  'id': "User's ID is 1",
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

class EnrichContent(PromptTemplate):
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
- For every field present (e.g., `subject`, `topic`), summarize it in natural language.
- Relate the content back to the user's needs or learning goals if applicable.
- Maintain the same keys as input.

Return the result as a JSON object. Do not include explanations outside the JSON.
"""
            ),
            input_variables=["action", "existing_data", 'current_resources_data']
        )

    def format_prompt(self, action: str, existing_data: dict, current_resources_data : dict) -> str:
        """
        Convert input data to formatted string and inject into prompt.
        """
        return self.format(
            action=action,
            existing_data=json.dumps(existing_data, indent=2),
            current_resources_data=json.dumps(current_resources_data, indent=2)
        )


class ContentGenerationTemplate(PromptTemplate):
    """
    Prompt to generate markdown educational content ONLY.

    The model must return ONLY the markdown content as a plain string.
    No JSON, no metadata, no explanations outside the content.
    """

    def __init__(self):
        super().__init__(
            template=(
                """You are an educational content generator.

Task: {action}

User Info:
{user_data}

Learning Resource:
{resource_data}

Instructions:
- Generate a clear, structured markdown lesson/explanation.
- Explain concepts in a way that feels like a friendly tutor.
- Focus on the subject and topic provided.
- Use headings, bullet points, and code blocks as needed.
- Ensure the content is educational and engaging.
- Tailor it to the user's grade level and interests.
- Return ONLY the markdown content text and dont introduce yourself or provide any other information.
- Do NOT return JSON, metadata, or extra commentary.
"""
            ),
            input_variables=["action", "user_data", "resource_data"]
        )

    def format_prompt(self, action: str, user_data: dict, resource_data: dict) -> str:
        return self.format(
            action=action,
            user_data=json.dumps(user_data, indent=2),
            resource_data=json.dumps(resource_data, indent=2)
        )


CONTENT_IMPROVISE_SYSTEM_PROMPT = SystemMessage(content="""
You are an energetic and insightful educational content improver and enhancer.

Your task:  
Take the given educational content and improve it by making it more engaging, clear, and reader-friendly while preserving the original meaning and key points.

Focus on:  
- Enhancing structure with clear markdown headings, bullet points, and examples.  
- Injecting a warm, professional, and approachable tone — friendly but not overly casual.  
- Adding vivid metaphors and real-world connections to make concepts memorable.  
- Improving flow and readability — make it easy to scan and digest.  
- Including occasional motivational nudges or thoughtful questions (1-2 per passage) that invite reflection and curiosity without overwhelming the reader.  
- Avoiding unnecessary repetition or filler language.  
- Explaining *why* topics matter, not just *what* they are.  
- Maintaining concise, clear language suitable for motivated learners who want efficient and deep understanding.  

**Important:**  
- Return ONLY the markdown content.  
- DO NOT return JSON, metadata, or any extra explanations.  

Example opening you might use to improve a draft:  
“Let’s dive into [subject] — understanding this will unlock powerful tools for your learning journey!”

Now, improve the following content:

""")

class RouteSelectorNode(PromptTemplate):
    def __init__(self):
        super.__init__(
            template = '''
            You are a route selector for an educational learning system.
            Your task is to determine the next action based on the user''s current state and progress.'''
    )




prompt_user = UserSummaryTemplate()
prompt_resource = LearningResourceTemplate()
prompt_content_generation = ContentGenerationTemplate()
prompt_content_improviser = CONTENT_IMPROVISE_SYSTEM_PROMPT
prompt_route_selector = RouteSelectorNode()

user_summary = (prompt_user | get_geminit_gemini_model(LearningResource))
user_content_generation = (prompt_content_model(UserInfo))
learning_resource = (prompt_resource | ge_generation | get_gemini_model(ContentResponse))
content_improviser = get_groq_model()

# Node references for graph construction or direct use
user_info = user_info_node
learning_resource = learning_resource_node
content_generation_node = content_generation
content_improviser = content_improviser_node
