import json
from functools import wraps


# Sometimes the LLM model is not able to output correctly a JSON string
def llm_chain_retry(max_retries):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    output = func(*args, **kwargs)
                    return json.loads(output)
                except json.JSONDecodeError:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1}/{max_retries}: Failed to decode JSON. LLM output JSON is malformed.")
            raise ValueError(f"Failed to decode JSON after {max_retries} attempts. LLM output JSON is malformed.")
        return wrapper
    return decorator
