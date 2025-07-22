from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import  ChatGroq
from keys.apis import set_env
from schemas import UserInfo

def get_gemini_model(output_schema):

    return ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        api_key=set_env('GOOGLE_API_KEY'),
        temperature=1,
    ).with_structured_output(output_schema)

def get_groq_model(output_schema):
    return ChatGroq(
        model='groq-llama-3.1-70b',
        api_key=set_env('GROQ_API_KEY'),
        temperature=0.5
    ).with_structured_output(output_schema)