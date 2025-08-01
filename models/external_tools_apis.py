import requests
import os

from langsmith import expect
from sqlalchemy.testing.plugin.plugin_base import logging

from keys.apis import set_env

def serp_api_tool(query:str) -> dict:
    """
    Use SerpAPI to search for the given query and return the results.
    Returns a dictionary with the search results.
    """

    try:
        api_key = set_env('SERP_API_KEY')
        if not api_key:
            raise ValueError("SERP_API_KEY is not set. Please set it in your environment variables.")
        params = {
            'api_key': api_key,
            'query': query,
            'engine': "google",
            "num": 5,
        }
        response = requests.get('https://serpapi.com/search', params=params)
        data = response.json()

    except Exception as e:
        logging.error(f"Error in SerpAPI tool: {e}")
    return data