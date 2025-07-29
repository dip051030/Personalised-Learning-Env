import os
import getpass
from typing import Optional
import logging
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

load_dotenv()


def set_env(var: str) -> Optional[str]:
    """
    Prompt for and set an environment variable if not already set.
    Returns the value of the environment variable.
    """
    if not os.environ.get(var):
        logging.info(f"Prompting for environment variable: {var}")
        os.environ[var] = getpass.getpass(f"Enter {var}: ")
    logging.info(f"Environment variable {var} set.")
    return os.environ.get(var)