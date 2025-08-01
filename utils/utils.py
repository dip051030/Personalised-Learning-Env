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
        if hasattr(state, 'model_dump'):
            state_data = state.model_dump()
        elif hasattr(state, 'dict'):
            state_data = state.dict()
        elif isinstance(state, dict):
            state_data = state
        else:
            raise ValueError("State object does not support serialization.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=4, ensure_ascii=False)
        logging.info(f"LearningState saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save LearningState to {file_path}: {e}")
