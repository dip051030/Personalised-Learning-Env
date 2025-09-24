import json
import logging
from pathlib import Path
from typing import List, Dict, Any


DATA_DIR = Path(__file__).parent.parent / "data"


def load_json_data(filename: str) -> List[Dict[str, Any]]:
    """
    Load lesson data from a JSON file in the lessons data directory.

    Args:
        filename (str): The name of the lesson data file.

    Returns:
        List[Dict[str, Any]]: List of lesson data dictionaries.

    Raises:
        FileNotFoundError: If the file does not exist in the data directory.
    """
    path = DATA_DIR / filename
    try:
        logging.info(f"INFO Loading lesson data from {path}")
        if not path.exists():
            logging.error(f"ERROR {filename} not found in {DATA_DIR}")
            raise FileNotFoundError(f"{filename} not found in {DATA_DIR}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"INFO Loaded {len(data)} lessons from {filename}")
    except FileNotFoundError as e:
        logging.error(f"ERROR File not found: {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"ERROR Failed to decode JSON from {filename}: {e}")
        return []
    except Exception as e:
        logging.error(f"ERROR An unexpected error occurred while loading lesson data: {e}")
