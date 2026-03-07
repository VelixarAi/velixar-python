import html
from typing import Any, Dict

def sanitize_html(text: str) -> str:
    return html.escape(text)

def sanitize_for_template(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for key, value in context_dict.items():
        if isinstance(value, str):
            sanitized[key] = html.escape(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_for_template(value)
        else:
            sanitized[key] = value
    return sanitized