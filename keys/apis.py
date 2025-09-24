import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def set_env(var: str) -> Optional[str]:
    """
    Retrieves an environment variable. Logs a warning if not set.
    Returns the value of the environment variable.
    """
    value = os.environ.get(var)
    if not value:
        logging.warning(f"Environment variable {var} is not set. Please ensure it is configured.")
    return value
