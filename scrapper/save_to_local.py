import json
from typing import Union

from models.external_tools_apis import serp_api_tool
from schemas import LearningState

def serper_api_results_parser(state: LearningState) -> dict:
    """
    Parses the results from the SerpAPI tool based on the current learning state.

    Args:
        state (LearningState): The current learning state containing the topic and grade.

    Returns:
        dict: The search results retrieved from the SerpAPI tool.
    """
    serpapi_search_results = serp_api_tool(
        query=state.current_resource.topic + 'for grade ' + str(state.current_resource.grade))
    return serpapi_search_results


def save_to_local(data: Union[dict, list], file_path: str):
    """
    Saves the provided data (dict or list) to a local JSON file.

    Args:
        data (dict or list): The data to be saved.
        file_path (str): The path where the data will be saved.

    Raises:
        TypeError: If the data contains unsupported types for JSON serialization.
    """
    if not isinstance(data, (dict, list)):
        raise TypeError("Data must be a dict or a list to be saved as JSON.")
    with open(file_path, mode='w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)