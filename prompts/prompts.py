from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
import json
import logging
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from models.llm_models import get_gemini_model, get_groq_model, get_deepseek_model
from schemas import UserInfo, LearningResource, LearningState, ContentResponse, EnrichedLearningResource, RouteSelector, FeedBack

class UserSummaryTemplate(PromptTemplate):
    """
    Template for summarizing user data. The LLM is instructed to return natural
    language descriptions for **each key** in the user object, including fields
    like `username`, `age`, `grade`, `is_active`, `subject`, and `topic`.

    It returns a JSON object with the same keys, but values are reworded explanations.
    """

    def __init__(self, *args, **kwargs):
        logging.info("Initializing UserSummaryTemplate")
        super().__init__(
            template=(
                """Your role is {action}. You are given structured user data: {existing_data}
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
            input_variables=["action", "existing_data"], *args, **kwargs
        )

    def format_prompt(self, action: str, existing_data: dict) -> str:
        """
        Convert the input dict to a pretty JSON string for better formatting and inject it into the prompt.
        """
        logging.info(f"Formatting UserSummaryTemplate prompt for action: {action}")
        return self.format(
            action=action,
            existing_data=json.dumps(existing_data, indent=2)
        )

# -----------------------------------------------------------------------------------------
# 📚 LearningResourceTemplate: Summarizes learning content + links it to user interest
# -----------------------------------------------------------------------------------------

class EnrichContent(PromptTemplate):
    def __init__(self):
        logging.info("Initializing EnrichContent Template")
        super().__init__(
            template="""
You are a curriculum enrichment agent. Your job is to {action}.

You have access to:
- Structured resource data:
{foundation_data}

- External search data:
{scrapped_data}

Instructions:
- Use both the structured data and the external search data to enrich any vague or brief fields in the resource.
- Make enhancements student-friendly and formal, using visual analogies, real-world applications, and clear explanations where appropriate.
- Preserve all original keys and the overall structure of the resource.
- Only enrich or expand existing fields; do not add new keys or sections.

Output:
Return a single, valid JSON object with the enriched resource. Do not include any explanations or extra text.
""",
            input_variables=["current_resources_data", "action", "titles", "snippets"]
        )


    def format_prompt(
        self,
        action: str,
        current_resources_data: dict,
        titles: List[str] = None,
        snippets: List[str] = None
    ) -> str:

        logging.info("Formatting EnrichContent prompt")

        return self.format(
            action=action,
            current_resources_data=json.dumps(current_resources_data, indent=2),
            titles="\n".join(titles),
            snippets="\n".join(snippets)
        )

class ContentGenerationTemplate(PromptTemplate):
    """
    Dynamic prompt to generate markdown educational content ONLY.
    The model must return ONLY the Markdown content as a plain string.
    No JSON, no metadata, no explanations outside the content.
    The structure and content adapt to user input (class, topic, curriculum, etc.).
    """

    def __init__(self):
        logging.info("Initializing ContentGenerationTemplate")
        super().__init__(
            template=(
"""You are an expert educational content generator.

Task: {action}

User Info:
{user_data}

Learning Resource:
{resource_data}

Style:
{style}

Instructions:
- Generate a clear, structured markdown lesson or explanation for the topic and class provided.
- Use the following structure, adapting each section to the user's class, curriculum, and interests:
- Follow the metadata strictly.
# Topic Title: [Use the topic from the action or user_data]

## Introduction
Briefly define and introduce the topic, tailored to the user's grade/class and curriculum.

## Real-Life Application
Give practical, relatable examples relevant to the user's context (e.g., local curriculum, age group).

## Formula & Explanation
Present any key equations, variable definitions, and derivations as appropriate for the user's level.

## Curriculum Relevance
Explain how this topic fits into the user's curriculum (e.g., NEB Class 12, CBSE Class 10, etc.).

## Frequently Asked Questions
List 2-3 common questions students at this level might ask, and answer them clearly.

## Summary
Summarize the lesson in 2-3 concise lines.

---

**Tags**: [Include class, subject, topic, curriculum, and any relevant keywords from user_data, resource_data, or style]

IMPORTANT: Output ONLY the markdown lesson content. Do NOT include any explanations, JSON, or extra text before or after the markdown.
"""
            ),
            input_variables=["action", "user_data", "resource_data", "style"]
        )

    def format_prompt(self, action: str, user_data: dict, resource_data: dict, style: str) -> str:
        logging.info(f"Formatting ContentGenerationTemplate prompt for action: {action}")
        return self.format(
            action=action,
            user_data=json.dumps(user_data, indent=2),
            resource_data=json.dumps(resource_data, indent=2),
            style=style
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
- Be sure to generate highly engaging content that resonates with the reader's interests and learning style.
- The rating should be more than what it was before, so you can improve the content.
- Use the gaps to generate a more engaging and informative content for the user.
**Important:**  
- Return ONLY the markdown content.  
- DO NOT return JSON, metadata, or any extra explanations.  

Example opening you might use to improve a draft:  
“Let’s dive into [subject] — understanding this will unlock powerful tools for your learning journey!”

Now, improve the following content:

""")

class BlogGenerationPrompt(PromptTemplate):
    """
    Template for generating educational blog posts.
    The LLM should:
    - Translate academic topic into engaging blog format
    - Make it informative but also fun/relatable
    - Consider user's interests and the style logic
    """

    def __init__(self):
        logging.info("Initializing BlogGenerationPrompt Template")
        super().__init__(
            template=(
"""You're a friendly education blogger.

USER PROFILE:
{user_data}

TOPIC INFORMATION:
{resource_data}

STYLE TO FOLLOW:
{style}

Write a short, engaging blog post for students based on the above topic.

- Make it informative but not too formal.
- Use real-world analogies and visuals if appropriate.
- Output Markdown-formatted blog content only."""
            ),
            input_variables=["user_data", "resource_data", "style"]
        )

    def format_prompt(self, user_data: dict, resource_data: dict, style: str) -> str:
        logging.info(f"Formatting BlogGenerationPrompt for style: {style}")
        return self.format(
            user_data=json.dumps(user_data, indent=2),
            resource_data=json.dumps(resource_data, indent=2),
            style=style
        )

class RouteSelectorNode(PromptTemplate):
    def __init__(self):
        logging.info("Initializing RouteSelectorNode Template")
        super().__init__(
            template = '''
You are a route selector for an educational learning system.
Your task is to determine the next action based on the user''s current state and progress.
Based on the {current_resources} decide whether to generate a blog or a lesson and return the output.'''
    , input_variables=["current_resources"]
        )

    def format_prompt(self, current_resources: dict) -> str:
        logging.info(f"Formatting RouteSelectorNode prompt for action.")
        return self.format(
            current_resources=json.dumps(current_resources, indent=2)
        )


CONTENT_FEEDBACK_SYSTEM_PROMPT = SystemMessage(content="""
You are an intelligent feedback assistant trained to process and structure user feedback on educational content.

Your goal is to analyze the provided content and comments, and generate a clean JSON object with the following fields ONLY:

- `rating`: An integer from 1 to 5 (1 = very poor, 5 = excellent).
- `comments`: A short criticised summary of the content.
- `needed`: A boolean indicating if feedback is needed (True) or not (False).
- `gaps`: A list of specific content gaps or areas for improvement (optional, can be empty).

**Instructions:**
- Return ONLY a valid JSON object with these four fields.
- Do NOT include any extra text, explanation, or fields.
- If a field is not available, still include it with a reasonable default (e.g., `comments` can be an empty string, `gaps` can be an empty list).
- Never invent new fields or explanations.
- Your output must be a single JSON object, nothing else.

Example output:
{
  "rating": 4,
  "comments": "The explanation was clear and engaging, but could use more real-world examples.",
  "needed": true,
  "gaps": [
    "Missing real-world applications of the concept.",
    "Could include more visual aids."
  ]
}
""")

class ContentGapGenerationPrompt(PromptTemplate):
    """
    Prompt to identify content gaps in educational material based on feedback and the content itself.
    The LLM should return a JSON object matching the FeedBack class, with:
      - rating: integer (1-5)
      - comments: summary of feedback and identified gaps
      - needed: boolean (True if improvement is needed)
      - gaps: list of specific content gaps (optional, for next model)
    """

    def __init__(self):
        super().__init__(
            template=(
"""You are an expert educational content reviewer.

Your task is to analyze the following learning content and the feedback provided, and identify any content gaps, missing explanations, unclear sections, or areas for improvement.

CONTENT:
{content}

FEEDBACK:
{feedback}

Instructions:
- Carefully read both the content and the feedback.
- Identify and list all content gaps, missing details, or unclear explanations.
- If the feedback already mentions gaps, include them. If you find additional gaps, add those too.
- Return a JSON object with the following fields ONLY:
  - "rating": integer from 1 to 5 (overall quality)
  - "comments": a short summary of the main feedback and gaps
  - "needed": true if improvement is needed, false if not
  - "gaps": a list of specific content gaps or improvement points

Example output:
{{
  "rating": 3,
  "comments": "The content is generally clear but lacks real-world examples and visual aids. Some sections are too brief.",
  "needed": true,
  "gaps": [
    "No real-world examples provided.",
    "Missing diagrams or visual explanations.",
    "The explanation of the formula derivation is too brief."
  ]
}}
- Do NOT include any extra text or explanation outside the JSON.
"""
            ),
            input_variables=["content", "feedback"]
        )

    def format_prompt(self, content: str, feedback: dict) -> str:
        return self.format(
            content=content,
            feedback=feedback
        )


prompt_user = UserSummaryTemplate()
prompt_enrichment = EnrichContent()
prompt_content_generation = ContentGenerationTemplate()
prompt_content_improviser = CONTENT_IMPROVISE_SYSTEM_PROMPT
prompt_feedback = CONTENT_FEEDBACK_SYSTEM_PROMPT
prompt_route_selector = RouteSelectorNode()
prompt_blog_generation = BlogGenerationPrompt()
prompt_gap_finder = ContentGapGenerationPrompt()

user_summary = prompt_user | get_gemini_model(UserInfo)
enriched_content = prompt_enrichment | get_gemini_model(EnrichedLearningResource)
route_selector = prompt_route_selector | get_gemini_model(RouteSelector)
content_generation = prompt_content_generation | get_gemini_model(ContentResponse)
blog_generation = prompt_blog_generation | get_gemini_model(ContentResponse)
gap_finder = prompt_gap_finder | get_gemini_model(FeedBack)
content_improviser =get_groq_model()
content_feedback = get_deepseek_model(FeedBack)
