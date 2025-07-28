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
    A simplified prompt template for enriching a learning resource.
    The model is expected to:
    - Expand and clarify fields like 'description' and 'elaboration'
    - Generate summaries and student-friendly explanations
    - Maintain the same keys as input (structured JSON output)
    """

    def __init__(self):
        super().__init__(
            template=(
                """You're a curriculum enrichment agent. Based on this structured topic:
{current_resources_data}

Your task:
- Enrich vague or brief fields (like `description`, `elaboration`)
- Add a student-friendly summary
- Include optional insights if relevant (e.g., practical uses, visual analogies)
- Keep original keys. Maintain consistent structure.

Return only a single valid JSON object. Do not explain your process.
"""
            ),
            input_variables=["current_resources_data"]
        )

    def format_prompt(self, current_resource_data: dict) -> str:
        """
        Converts the provided topic data dictionary into a JSON-formatted
        string to insert into the prompt.
        """
        return self.format(
            current_resource_data=json.dumps(current_resource_data, indent=2)
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
            Your task is to determine the next action based on the user''s current state and progress.
            Based on the {current_resources} decide whether to generate a blog or a lesson and return the output.'''
    )


    def format_prompt(self, action: str, current_resources_data: dict) -> str:
        return self.format(
            action=action,
            current_resources_data=json.dumps(current_resources_data, indent=2)
        )



prompt_user = UserSummaryTemplate()
prompt_enrichment = EnrichContent()
prompt_content_generation = ContentGenerationTemplate()
prompt_content_improviser = CONTENT_IMPROVISE_SYSTEM_PROMPT
prompt_route_selector = RouteSelectorNode()

user_summary = (prompt_user | get_geminit_gemini_model(LearningResource))
user_content_generation = (prompt_content_model(UserInfo))
enriched_content = (prompt_enrichment | get_gemini_model(EnrichContent))
content_improviser = get_groq_model()