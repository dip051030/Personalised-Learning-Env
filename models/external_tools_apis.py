import requests
import logging
from keys.apis import set_env


def serp_api_tool(query: str) -> dict:
    """
    Use SerpAPI to search for the given query and return the results.
    Returns a dictionary with the search results.
    """
    data = {}
    try:
        api_key = set_env('SERP_API_KEY')
        if not api_key:
            raise ValueError("SERP_API_KEY is not set. Please set it in your environment variables.")
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        params = {
            'q': query + ' site:byjus.com OR site:vedantu.com OR site:toppr.com OR site:learnfatafat.com',
            'engine': "google",
            'num': 10,
            'gl': 'in'
        }
        response = requests.post('https://google.serper.dev/search', json=params, headers=headers)
        data = response.json()
        logging.info(f"[external_tools_apis.py:{serp_api_tool.__code__.co_firstlineno}] INFO SerpAPI request successful for query: {query}")
    except Exception as e:
        logging.error(f"[external_tools_apis.py:{serp_api_tool.__code__.co_firstlineno}] ERROR in SerpAPI tool: {e}")
        data = {"error": str(e)}

    return data