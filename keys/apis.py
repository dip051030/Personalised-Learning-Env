import os
import getpass
from typing import Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def set_env(var: str) -> Optional[str]:
    if not os.environ.get(var):
        logging.info(f"Prompting for environment variable: {var}")
        os.environ[var] = getpass.getpass(f"Enter {var}: ")
    logging.info(f"Environment variable {var} set.")
    return os.environ.get(var)