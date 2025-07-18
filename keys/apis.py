import os
import getpass
from typing import Optional


def set_env(var: str) -> Optional[str]:
    """Safely sets an environment variable if not already set.
    Args:
        var: Environment variable name
    Returns:
        The set value or None if already exists
    """
    if not os.environ.get(var):
        # Get input securely
        value = getpass.getpass(f"Enter value for {var}: ").strip()

        # Only set if user actually entered something
        if value:
            os.environ[var] = value
            return value
        print(f"Warning: Empty value provided for {var}")
    return None