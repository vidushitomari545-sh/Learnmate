import os
import re

from app import config
from openai import OpenAI


def sanitize_error(message: str) -> str:
    sanitized = message
    for value in sorted(os.environ.values(), key=len, reverse=True):
        if value:
            sanitized = sanitized.replace(value, "[REDACTED]")

    sanitized = re.sub(
        r"(?is)\bheaders?\s*[:=]\s*(?:\{.*?\}|\[.*?\]|\(.*?\))",
        "headers=[REDACTED]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?is)\brequest(?:\s+object)?\s*[:=].*$",
        "request=[REDACTED]",
        sanitized,
    )
    sanitized = re.sub(
        r"(?i)(authorization|api[-_ ]?key|openai_api_key)"
        r"(['\"]?\s*[:=]\s*['\"]?)[^,\s'\"}]+",
        r"\1\2[REDACTED]",
        sanitized,
    )
    sanitized = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [REDACTED]", sanitized)
    sanitized = re.sub(r"\bsk-[A-Za-z0-9_-]+\b", "[REDACTED]", sanitized)
    return " ".join(sanitized.split())[:500]


try:
    if config.LLM_PROVIDER != "openai":
        raise ValueError("Configured provider is not openai")
    if config.LLM_MODEL != "gpt-6-luna":
        raise ValueError("Configured model is not gpt-6-luna")

    response = OpenAI(
        timeout=config.LLM_TIMEOUT_SECONDS,
        max_retries=0,
    ).responses.create(
        model="gpt-6-luna",
        input="Reply with exactly: LearnMate is connected to the real OpenAI API.",
        max_output_tokens=32,
    )
    output = response.output_text
    if not output:
        raise RuntimeError("The API returned no response text")

    print(output)
    print("SUCCESS")
except Exception as exc:
    print(f"{type(exc).__name__}: {sanitize_error(str(exc))}")
    raise SystemExit(1)
