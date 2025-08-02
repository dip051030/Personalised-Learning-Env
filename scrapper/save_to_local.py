import json
serpapi_search_results = {}
def save_to_local(data, file_path:str):
    """
    Save the learning state to a local JSON file.

    Args:
        file_path (str): The path where the learning state will be saved.
    """
    with open(file_path, mode='w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


save_to_local(serpapi_search_results, 'serpapi_search_results.json')
