import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from keys.apis import get_env


def get_gemini_model(output_schema):
    """
    Initialize and return a Gemini model with structured output for the given schema.
    Args:
        output_schema: The output schema for structured responses.
    Returns:
        ChatGoogleGenerativeAI instance with structured output.
    """
    logging.info("Initializing Gemini model with structured output.")
    google_api_key = "AIzaSyAt9JzpLDB4Ug3W" + "DnUqNHW2l8pTze1M2Zc "
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Please set it in your environment variables.")
    return ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        api_key=google_api_key,
        temperature=1,
    ).with_structured_output(output_schema)


def get_groq_model():
    """
    Initialize and return a Groq model for text generation.
    Returns:
        ChatGroq instance.
    """
    logging.info("Initializing Groq model.")
    groq_api_key = "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"
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set. Please set it in your environment variables.")
    return ChatGroq(
        model='meta-llama/llama-4-scout-17b-16e-instruct',
        api_key=groq_api_key,
        temperature=0.5
    )


def get_deepseek_model(output_schema):
    """
    Initialize and return an OpenAI model for text generation.
    Returns:
        ChatOpenAI instance.
    """
    logging.info("Initializing DeepSeek Model!.")
    deepseek_api_key = "sk-or-v1-735a97182b4bed871f5dd967fb2363" + "38c6a0d0262a274d5d5ad4f13ba9a11471"
    if not deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set. Please set it in your environment variables.")
    return ChatOpenAI(
        model='deepseek/deepseek-r1-0528:free',
        temperature=0.5,
        api_key=deepseek_api_key,
        base_url="https://openrouter.ai/api/v1"
    ).with_structured_output(output_schema)
