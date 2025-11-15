import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def get_env(var: str, raise_error: bool = False) -> Optional[str]:
    """
    Retrieves an environment variable. Logs a warning if not set.
    If `raise_error` is True, raises a ValueError if the variable is not set.
    Returns the value of the environment variable.
    """
    value = os.environ.get(var)
    if not value:
        message = f"Environment variable {var} is not set. Please ensure it is configured."
        if raise_error:
            raise ValueError(message)
        else:
            logging.warning(message)
    return value
