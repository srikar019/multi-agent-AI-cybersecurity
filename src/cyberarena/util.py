"""Small shared helpers."""
from __future__ import annotations

import json
import re


def message_text(msg: object) -> str:
    """Normalize an LLM response's .content to a plain string.

    Some providers (e.g. Google Gemini via langchain-google-genai 4.x) return
    content as a list of content parts rather than a single string.
    """
    content = getattr(msg, "content", None)
    if content is None:
        content = str(msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or str(item))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "\n".join(parts)
    return str(content)


def extract_json(text: str) -> object:
    """Extract the first complete JSON value (object or array) from text.

    Tolerates markdown fences, prose, and trailing content.
    """
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.S)
    text = re.sub(r"\s*```$", "", text, flags=re.S)

    # Find whichever JSON container opens first ({...} or [...]).
    start_o = text.find("{")
    start_a = text.find("[")
    if start_o == -1 and start_a == -1:
        raise ValueError("no JSON found in model output")
    if start_o == -1:
        start = start_a
    elif start_a == -1:
        start = start_o
    else:
        start = min(start_o, start_a)

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON in model output")