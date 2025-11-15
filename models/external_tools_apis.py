import logging

import requests


def serp_api_tool(query: str) -> dict:
    """
    Use SerpAPI to search for the given query and return the results.
    Returns a dictionary with the search results.
    """
    data = {}
    try:
        api_key = "e8cf5cfb5d8b59d957" + "c3f6576779c499ca177287"
        if not api_key:
            raise ValueError("SERP_API_KEY is not set. Please set it in your environment variables.")
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        params = {
            'q': query + ' site:byjus.com OR site:toppr.com OR site:learnfatafat.com',
            'engine': "google",
            'num': 20,
            'gl': 'in'
        }
        response = requests.post('https://google.serper.dev/search', json=params, headers=headers)
        data = response.json()
        logging.info(f"INFO SerpAPI request successful for query: {query}")
    except requests.exceptions.RequestException as e:
        logging.error(f"ERROR Network or request error in SerpAPI tool: {e}")
        data = {"error": f"Network or request error: {e}"}
    except ValueError as e:
        logging.error(f"ERROR Configuration error in SerpAPI tool: {e}")
        data = {"error": f"Configuration error: {e}"}
    except json.JSONDecodeError as e:
        logging.error(f"ERROR JSON decoding error in SerpAPI tool: {e}")
        data = {"error": f"JSON decoding error: {e}"}
    except Exception as e:
        logging.error(f"ERROR An unexpected error occurred in SerpAPI tool: {e}")
        data = {"error": f"An unexpected error occurred: {e}"}

    return data
