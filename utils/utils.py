"""
Utility functions for saving learning state and generated content to files.
"""
import os
import json
import logging

def save_learning_state_to_json(state, file_path):
    """
    Save the details of the LearningState object or dict to a JSON file.
    If the file does not exist, it will be created.
    Args:
        state: LearningState object (should have .model_dump() or .dict() method) or dict
        file_path: Path to the JSON file
    """
    try:
        # Always use model_dump(mode="json") for full serialization of nested models
        if hasattr(state, 'model_dump'):
            state_data = state.model_dump(mode="json")
        elif hasattr(state, 'dict'):
            state_data = state.dict()
        elif isinstance(state, dict):
            state_data = state
        else:
            raise ValueError("State object does not support serialization.")
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=4, ensure_ascii=False)
        logging.info(f"[utils.py:{save_learning_state_to_json.__code__.co_firstlineno}] INFO LearningState saved to {file_path}")
    except Exception as e:
        logging.error(f"[utils.py:{save_learning_state_to_json.__code__.co_firstlineno}] ERROR Failed to save LearningState to {file_path}: {e}")

def save_generated_content(content, file_path):
    """
    Save the generated content (string) to a separate file.
    If the file's directory does not exist, it will be created.
    Args:
        content: The generated content as a string.
        file_path: Path to the file where content will be saved.
    """
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logging.info(f"[utils.py:{save_generated_content.__code__.co_firstlineno}] INFO Generated content saved to {file_path}")
    except Exception as e:
        logging.error(f"[utils.py:{save_generated_content.__code__.co_firstlineno}] ERROR Failed to save generated content to {file_path}: {e}")

def read_from_local(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"[utils.py:{read_from_local.__code__.co_firstlineno}] INFO Data read from {file_path}")
    urls = [item.get('link') for item in data if isinstance(item, dict) and 'link' in item]
    return urls

