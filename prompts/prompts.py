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
from schemas import UserInfo, LearningResource, LearningState, ContentResponse, EnrichedLearningResource, RouteSelector, \
    FeedBack, PostValidationResult


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
- Structured resource data (this is the base object you must work from):
{foundation_data}

- External search data (for enrichment only):
{scrapped_data}

Instructions:
- Begin by copying the structured resource data **exactly** as your starting point.
- For each field in the base data:
    - If the field is vague, brief, or unclear, use the external search data to expand, clarify, or enrich it.
    - If the field is already detailed and clear, leave it unchanged.
- Do **not** remove, rename, reorder, or omit any fields from the base data, even if they are empty or not enriched.
- Do **not** add new fields or sections; only update/enrich existing fields.
- Always preserve all original keys and the overall structure of the resource, including fields like quiz, case_study, hands_on_demo, simulation_hints, etc.
- If a field does not need enrichment, copy it exactly as in the base data.
- If the base data includes fields such as simulation_hints or quiz, add technical finishing touches such as simulation links, quiz answers, or other relevant details to make the resource more complete and actionable.
- The output JSON **must have exactly the same keys and structure as the base data**.

Output:
Return a single, valid JSON object with the enriched resource. Do not include any explanations or extra text.
""",
            input_variables=["action", "foundation_data", "scrapped_data"]
        )

    def format_prompt(
        self,
        action: str,
        foundation_data: dict,
        scrapped_data: dict
    ) -> str:
        logging.info("Formatting EnrichContent prompt")
        return self.format(
            action=action,
            foundation_data=json.dumps(foundation_data, indent=2),
            scrapped_data=json.dumps(scrapped_data, indent=2)
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
"""You are an expert educational content writer.

Objective: {action}

User Context:
{user_data}

Learning Resource Metadata:
{resource_data}

Preferred Style and Tone:
{style}

Instructions:
- Generate a well-structured, markdown-formatted educational lesson tailored to the specified class and topic.
- Use the exact structure below, adjusting each section for the student's level, curriculum, and background.

Structure to Follow:
# Topic Title  
(Use the exact topic mentioned in {action} or {resource_data}. Keep it informative, not poetic or metaphorical.)

## Introduction  
Introduce and define the topic clearly, using terminology and examples suitable for the student's grade and curriculum.

## Real-Life Application  
Provide 1–2 practical examples or case studies that relate the topic to everyday life or local context (e.g., technology, environment, social relevance).

## Formula & Explanation  
Include important formulas (if any), define each variable, and explain the derivation or reasoning behind the formula as per the student’s level.

## Curriculum Relevance  
Briefly describe where this topic fits in the curriculum (e.g., "NEB Class 12 Physics Unit 5") and its importance in exams or concepts that follow.

## Frequently Asked Questions  
Include 2–3 common student questions related to this topic, and answer them in a simple, concise way.

## Summary  
Wrap up the key points of the topic in 2–3 short lines.

---

**Tags**: [Include keywords such as class, subject, topic name, curriculum board, and any other identifiers from {user_data} or {resource_data}]

OUTPUT FORMAT: Markdown only. Do NOT include explanations, JSON wrappers, or commentary before or after the markdown.

"""),
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
    - Translate an academic topic into an engaging blog format
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

Your task is to analyze the provided content and comments, and generate a valid JSON object with **exactly** the following four fields:

- `rating`: An integer from 1 to 5 (1 = very poor, 5 = excellent).
- `comments`: A concise, critical summary of the content.
- `needed`: A boolean indicating if feedback is needed (`true`) or not (`false`).
- `gaps`: A list of specific content gaps or areas for improvement (may be empty).
- `ai_reliability_score`: A float between 0 and 1 indicating the reliability of the AI-generated content.

**Strict instructions:**
- Output **only** a valid JSON object with these four fields, and nothing else.
- Do **not** include any extra text, explanations, or additional fields.
- If a field is missing or unavailable, include it with a default value (`comments` as an empty string, `gaps` as an empty list).
- Do **not** invent or add any new fields, keys, or explanations.
- The output must be a single, valid JSON object, and must not be wrapped in markdown, code blocks, or any other formatting.

**Example output:**
{
  "rating": 4,
  "comments": "The explanation was clear and engaging, but could use more real-world examples.",
  "needed": true,
  "gaps": [
    "Missing real-world applications of the concept.",
    "Could include more visual aids." 
  ],
  'ai_reliability_score': 0.85
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
- "ai_reliability_score": a float between 0 and 1 indicating the reliability of the AI-generated content.

Example output:
{{
"rating": 3,
"comments": "The content is generally clear but lacks real-world examples and visual aids. Some sections are too brief.",
"needed": true,
"gaps": ["No real-world examples provided.", "Missing diagrams or visual explanations.", "The explanation of the formula derivation is too brief."],
"ai_reliability_score": 0.75
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


CONTENT_SEO_OPTIMIZATION_SYSTEM_PROMPT = SystemMessage(content="""
You are a professional SEO editor and blog optimizer with expertise in educational content.
Your task is to improve the structure, formatting, and SEO metadata of the given markdown blog post. Do NOT generate new content, paraphrase, or alter the factual content. The content is already complete and must be preserved exactly as-is.
 
Apply only the following enhancements:
STRUCTURE RULES
- You should work on the provided markdown content only.
- Keep the H1 exactly as-is.
- Organize the blog using proper H2 and H3 headers.
- Break large paragraphs into smaller, scannable blocks.
- Use bullet points or numbered lists if appropriate — only on existing sentence-based content.
- Use valid markdown formatting (e.g., **bold**, #, ##, -) for layout clarity.

SEO OPTIMIZATION RULES
- Insert the primary keyword naturally into one subheading if not already present.
- Add a title_tag (maximum 60 characters) based on the H1.
- Add a meta_description (maximum 155 characters) summarizing the blog, using the primary keyword.
- Place the primary keyword in the first 25 words only if already found. Do not hallucinate.
- Add semantic structure by formatting only — do not alter wording.

E-E-A-T COMPLIANCE
- Preserve all original definitions, formulas, explanations, and terminology.
- Do NOT rephrase scientific or factual statements.
- Do NOT introduce new summaries or content.
- Do NOT invent citations, statistics, or call-to-actions.

STRICT EDITING RULES
- Do NOT rewrite, reword, or summarize.
- Do NOT add or remove any paragraph.
- Do NOT modify the tone, examples, or factual depth.
- Do NOT hallucinate any content, metadata, or enhancements.

#REMAINDER
- Return the **exact** markdown content with the enhancements applied.
- Don't loose the original content structure.
"""
)

POST_VALIDATION_SYSTEM_PROMPT = SystemMessage(content="""
You are an expert QA validator for SEO-optimized educational blog posts. Your job is to review the finalized markdown blog post and strictly validate that it follows all SEO, content integrity, and E-E-A-T guidelines.
Do NOT make improvements or suggestions. Simply detect and report any issues.

Validation Criteria:
1. Structural & Format Checks
- Does the blog have one and only one H1?
- Are H2 and H3 used meaningfully to organize content?
- Are large paragraphs broken down into smaller, scannable blocks?
- Are bullet points used appropriately (only when they were originally sentence-based)?
- Does the markdown formatting follow standard syntax?

2. SEO Compliance
- Is the title_tag under 60 characters?
- Is the meta_description under 155 characters and aligned with the blog content?
- Is the primary keyword present in the first 25 words (only if it existed in the original)?
- Is the keyword used in at least one subheading?

3. Content Integrity
- Are all original paragraphs and facts intact?
- Has any content been paraphrased, rewritten, or removed? (Should NOT be)
- Are all definitions, formulas, and explanations preserved exactly?
- Was any hallucinated content introduced? (e.g., links, tips, summaries, data, stats, citations)


Failure Conditions:
- Any factual alteration
- Any hallucinated addition (e.g., extra info, citations, summaries)
- Any missing original paragraph or heading
- Any AI-generated rewriting
- Any keyword stuffing or SEO manipulation

Output Format:

Return a JSON object in the format:

{
  "is_valid": true or false,
  "violations": [
    "<Clear description of the problem found, if any>"
  ]
}

If everything is perfect, return:

{
  "is_valid": true,
  "violations": []
}
""")

prompt_user = UserSummaryTemplate()
prompt_enrichment = EnrichContent()
prompt_content_generation = ContentGenerationTemplate()
prompt_content_improviser = CONTENT_IMPROVISE_SYSTEM_PROMPT
prompt_feedback = CONTENT_FEEDBACK_SYSTEM_PROMPT
prompt_seo_optimization = CONTENT_SEO_OPTIMIZATION_SYSTEM_PROMPT
prompt_route_selector = RouteSelectorNode()
prompt_blog_generation = BlogGenerationPrompt()
prompt_gap_finder = ContentGapGenerationPrompt()
prompt_post_validation = POST_VALIDATION_SYSTEM_PROMPT

user_summary = prompt_user | get_gemini_model(UserInfo)
enriched_content = prompt_enrichment | get_gemini_model(EnrichedLearningResource)
route_selector = prompt_route_selector | get_gemini_model(RouteSelector)
content_generation = prompt_content_generation | get_gemini_model(ContentResponse)
blog_generation = prompt_blog_generation | get_gemini_model(ContentResponse)
gap_finder = prompt_gap_finder | get_gemini_model(FeedBack)
content_seo_optimization = get_groq_model()
content_improviser =get_groq_model()
content_feedback = get_deepseek_model(FeedBack)
post_validation = get_deepseek_model(PostValidationResult)