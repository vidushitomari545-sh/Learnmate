import sys

from google import genai
from google.genai import types

from app import config

PROMPT = "Reply with exactly: LearnMate is connected to real Gemini AI."
MODEL = "gemini-3.6-flash"


def _attribute(value: object, name: str) -> object | None:
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _has_attribute(value: object, name: str) -> bool:
    try:
        return hasattr(value, name)
    except Exception:
        return False


def _enum_label(value: object) -> str:
    if value is None:
        return "none"
    name = _attribute(value, "name")
    if isinstance(name, str):
        return name
    return "unrecognized"


def _print_empty_response_diagnostics(response: object) -> None:
    prompt_feedback = _attribute(response, "prompt_feedback")
    block_reason = _attribute(prompt_feedback, "block_reason")
    candidates = _attribute(response, "candidates")
    response_has_candidates_field = _has_attribute(response, "candidates")
    candidates_are_list = isinstance(candidates, list)
    candidates_present = candidates_are_list and bool(candidates)

    print(f"candidates_present={'yes' if candidates_present else 'no'}")
    print(f"prompt_feedback_present={'yes' if prompt_feedback is not None else 'no'}")
    print(f"prompt_block_reason={_enum_label(block_reason)}")

    candidate_blocked = False
    has_readable_candidate = True

    def print_safety_ratings(prefix: str, ratings: object) -> None:
        nonlocal candidate_blocked
        if not isinstance(ratings, list):
            return
        for rating_index, rating in enumerate(ratings, start=1):
            category = _enum_label(_attribute(rating, "category"))
            blocked = _attribute(rating, "blocked")
            print(
                f"{prefix}_safety_{rating_index}="
                f"category:{category},"
                f"blocked:{'yes' if blocked is True else 'no' if blocked is False else 'unknown'}"
            )
            candidate_blocked = candidate_blocked or blocked is True

    print_safety_ratings(
        "prompt_feedback",
        _attribute(prompt_feedback, "safety_ratings"),
    )

    if candidates_are_list:
        for index, candidate in enumerate(candidates, start=1):
            finish_reason = _attribute(candidate, "finish_reason")
            finish_label = _enum_label(finish_reason)
            print(f"candidate_{index}_finish_reason={finish_label}")

            content = _attribute(candidate, "content")
            parts = _attribute(content, "parts")
            parts_are_list = isinstance(parts, list)
            text_parts_present = (
                parts_are_list
                and any(isinstance(_attribute(part, "text"), str) for part in parts)
            )
            print(
                f"candidate_{index}_content_present="
                f"{'yes' if content is not None else 'no'}"
            )
            print(
                f"candidate_{index}_parts_present="
                f"{'yes' if parts_are_list and parts else 'no'}"
            )
            print(
                f"candidate_{index}_text_parts_present="
                f"{'yes' if text_parts_present else 'no'}"
            )
            if not _has_attribute(candidate, "content") or not _has_attribute(
                candidate, "finish_reason"
            ):
                has_readable_candidate = False

            print_safety_ratings(
                f"candidate_{index}",
                _attribute(candidate, "safety_ratings"),
            )

            candidate_blocked = candidate_blocked or finish_label in {
                "SAFETY",
                "BLOCKLIST",
                "PROHIBITED_CONTENT",
                "SPII",
                "IMAGE_SAFETY",
                "IMAGE_PROHIBITED_CONTENT",
            }

    model_version = _attribute(response, "model_version")
    print(
        "model_version="
        + (
            model_version[:100]
            if isinstance(model_version, str)
            else "unavailable"
        )
    )
    usage = _attribute(response, "usage_metadata")
    usage_fields = (
        "prompt_token_count",
        "response_token_count",
        "tool_use_prompt_token_count",
        "thoughts_token_count",
        "total_token_count",
    )
    if usage is None:
        print("usage_metadata=unavailable")
    else:
        counts = []
        for field in usage_fields:
            count = _attribute(usage, field)
            if isinstance(count, int) and not isinstance(count, bool):
                counts.append(f"{field}:{count}")
        print(f"usage_metadata={','.join(counts) if counts else 'token counts unavailable'}")

    if block_reason is not None:
        category = "prompt-blocked"
    elif candidate_blocked:
        category = "candidate-blocked"
    elif not response_has_candidates_field:
        category = "unexpected-response"
    elif candidates is None or candidates == []:
        category = "missing-candidate"
    elif not candidates_are_list or not has_readable_candidate:
        category = "unexpected-response"
    else:
        category = "empty-content"
    print(f"FAILURE: {category}: Gemini returned no response text.")


def report_failure(error: Exception) -> None:
    error_name = type(error).__name__.lower()
    status_code = getattr(error, "code", getattr(error, "status_code", None))
    safe_status = (
        status_code
        if isinstance(status_code, int) and not isinstance(status_code, bool)
        else None
    )

    if "timeout" in error_name:
        category, message = "timeout", "Gemini request timed out."
    elif "authentication" in error_name or status_code in (401, 403):
        category, message = "authentication", "Gemini authentication failed."
    elif "ratelimit" in error_name or status_code == 429:
        category, message = "rate-limit", "Gemini rate limit or quota reached."
    elif "connection" in error_name or (
        isinstance(status_code, int) and status_code >= 500
    ):
        category, message = (
            "transient",
            "Gemini service is temporarily unavailable."
            + (f" HTTP status {safe_status}." if safe_status is not None else ""),
        )
    else:
        category, message = "provider", "Gemini smoke test failed."

    print(f"FAILURE: {category}: {message}")


if config.LLM_PROVIDER != "gemini":
    print("FAILURE: configuration: provider must be gemini.")
    raise SystemExit(1)
if config.GEMINI_MODEL != MODEL:
    print("FAILURE: configuration: model must be gemini-3.6-flash.")
    raise SystemExit(1)
if not config.GEMINI_API_KEY:
    print("FAILURE: configuration: GEMINI_API_KEY is missing.")
    raise SystemExit(1)

try:
    client = genai.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=types.HttpOptions(
            timeout=config.LLM_TIMEOUT_SECONDS * 1000,
            retry_options=None,
        ),
    )
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=PROMPT,
        config=types.GenerateContentConfig(
            max_output_tokens=128,
            thinking_config=types.ThinkingConfig(
                thinking_level=types.ThinkingLevel.MINIMAL,
            ),
        ),
    )
    try:
        response_text = response.text
    except Exception:
        response_text = None
    if not isinstance(response_text, str) or not response_text.strip():
        _print_empty_response_diagnostics(response)
        raise SystemExit(1)
    print(response_text)
    print("SUCCESS")
except SystemExit:
    raise
except Exception as error:
    report_failure(error)
    raise SystemExit(1)
