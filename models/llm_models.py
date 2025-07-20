from langchain_google_genai import ChatGoogleGenerativeAI
from keys.apis import set_env
from schemas import UserInfo

def get_llm_model(output_schema):

    return ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        api_key=set_env('GOOGLE_API_KEY'),
    ).with_structured_output(output_schema)