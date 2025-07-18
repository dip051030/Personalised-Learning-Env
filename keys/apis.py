import os
import getpass
from typing import Optional


def set_env(var: str) -> Optional[str]:
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Enter {var}: ")
    return os.environ.get(var)