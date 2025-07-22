from langchain_google_genai import ChatGoogleGenerativeAI
from lanchain_chat_groq import  ChatGroq
from keys.apis import set_env
from schemas import UserInfo

def get_gemini_model(output_schema):

    return ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        api_key=set_env('GOOGLE_API_KEY'),
    ).with_structured_output(output_schema)

