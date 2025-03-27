from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from pydantic import BaseModel, Field
from functools import wraps
import unicodedata
import xml.etree.ElementTree as ET


def remove_accents(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def convert_pydantic_to_bedrock_tool(model: BaseModel, description = None) -> dict:
    """
    Converts a Pydantic model to a tool description for the Amazon Bedrock Converse API.
    
    Args:
        model: The Pydantic model class to convert
        description: Optional description of the tool's purpose

    Returns:
        Dict containing the Bedrock tool specification        
    """
    # Validate input model
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise ValueError("Input must be a Pydantic model class")
    
    name = model.__name__
    input_schema = model.model_json_schema()
    tool = {
        'toolSpec': {
            'name': name,
            'description': description or f"{name} Tool",
            'inputSchema': {'json': input_schema }
        }
    }
    return tool


def load_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_all_keys(d: dict, keys=None) -> set:
    if keys is None:
        keys = set()
    if isinstance(d, dict):
        for k, v in d.items():
            keys.add(k)
            get_all_keys(v, keys)
    elif isinstance(d, list):  # In case there are lists of dicts
        for item in d:
            get_all_keys(item, keys)
    return keys


def retry_with_logging(max_attempts=3, wait_time=2):
    """
    A decorator that adds retry logic with detailed logging for ValueError.
    
    Args:
        max_attempts (int): Maximum number of retry attempts. Defaults to 3.
        wait_time (int): Seconds to wait between retry attempts. Defaults to 2.
    
    Returns:
        Decorated function with retry and logging capabilities.
    """
    def decorator(func):
        @wraps(func)
        @retry(
            stop=stop_after_attempt(max_attempts), 
            wait=wait_fixed(wait_time), 
            retry=retry_if_exception_type((ValueError, AssertionError)),
            before_sleep=lambda retry_state: print(
            f"Retry attempt {retry_state.attempt_number}: "
            f"Retrying {func.__name__} due to {type(retry_state.outcome.exception()).__name__}. "
            f"Reason: {retry_state.outcome.exception()}"
            )
        )
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


def retrieve_key_from_xml(xml_string: str, xml_key: str = 'respuesta') -> str:
    try:
        root = ET.fromstring(f"<root>{xml_string}</root>")  # Wrap in a root tag to handle multiple top-level elements
        answer_element = root.find(xml_key)
        return answer_element.text.strip() if answer_element is not None else None
    except Exception as e:
        raise e
    # except ET.ParseError:
    #     return None  # Handle invalid XML cases

