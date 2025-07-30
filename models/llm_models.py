import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from keys.apis import set_env

def get_gemini_model(output_schema):
    """
    Initialize and return a Gemini model with structured output for the given schema.
    Args:
        output_schema: The output schema for structured responses.
    Returns:
        ChatGoogleGenerativeAI instance with structured output.
    """
    logging.info("Initializing Gemini model with structured output.")
    return ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        api_key=set_env('GOOGLE_API_KEY'),
        temperature=1,
    ).with_structured_output(output_schema)

def get_groq_model():
    """
    Initialize and return a Groq model for text generation.
    Returns:
        ChatGroq instance.
    """
    logging.info("Initializing Groq model.")
    return ChatGroq(
        model='meta-llama/llama-4-scout-17b-16e-instruct',
        api_key=set_env('GROQ_API_KEY'),
        temperature=0.5
    )

def get_openai_model():
    """
    Initialize and return an OpenAI model for text generation.
    Returns:
        ChatOpenAI instance.
    """
    logging.info("Initializing OpenAI model.")
    return ChatOpenAI(
        model='gpt-3.5-turbo',
        temperature=0.5,
        api_key=set_env('OPENAI_API_KEY')
    )