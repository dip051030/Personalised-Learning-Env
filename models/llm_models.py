import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_groq import  ChatGroq
from keys.apis import set_env
from schemas import UserInfo

def get_gemini_model(output_schema):
    logging.info("Initializing Gemini model with structured output.")
    return ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        api_key=set_env('GOOGLE_API_KEY'),
        temperature=1,
    ).with_structured_output(output_schema)

def get_groq_model():
    logging.info("Initializing Groq model.")
    return ChatGroq(
        model='meta-llama/llama-4-scout-17b-16e-instruct',
        api_key=set_env('GROQ_API_KEY'),
        temperature=0.5
    )

def get_openai_model():
    logging.info("Initializing OpenAI model.")
    return ChatOpenAI(
        model='gpt-4o',
        temperature=0.5,
        api_key=set_env('OPENAI_API_KEY')
    )