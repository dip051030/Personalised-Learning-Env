import json
from pathlib import Path
from typing import List, Dict, Any

DATA_DIR = Path(__file__).parent.parent / "data" / "lessons"


def load_lesson_data(filename: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} not found in {DATA_DIR}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data
