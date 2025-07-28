import json
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DATA_DIR = Path(__file__).parent.parent / "data" / "lessons"


def load_lesson_data(filename: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / filename
    logging.info(f"Loading lesson data from {path}")
    if not path.exists():
        logging.error(f"{filename} not found in {DATA_DIR}")
        raise FileNotFoundError(f"{filename} not found in {DATA_DIR}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} lessons from {filename}")
    return data
