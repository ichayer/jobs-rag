import os
import sys
import warnings
from dotenv import load_dotenv, find_dotenv


def check_python_version():
    if sys.version_info[:2] != (3, 10):
        raise Exception("This code is only compatible with Python 3.10")

def load_env_vars(required_vars):
    load_dotenv(find_dotenv())
    env_vars = {var: os.environ.get(var) for var in required_vars}

    missing_vars = [var for var, val in env_vars.items() if not val]
    if missing_vars:
        raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}")

    return env_vars

# Ignore LanguageChain deprecated warnings
def suppress_warnings():
    warnings.filterwarnings("ignore")
