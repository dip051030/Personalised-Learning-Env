from langchain_google_genai import ChatGoogleGenerativeAI
from keys.apis import set_env

MODEL = ChatGoogleGenerativeAI(
    model = 'google-2.0-flash',
    api_key = set_env('GOOGLE_API_KEY'),
)