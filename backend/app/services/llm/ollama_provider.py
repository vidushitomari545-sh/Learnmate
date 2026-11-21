"""Local Ollama adapter for the provider-neutral LLM contract."""

import json
import socket
import time
import urllib.error
from collections.abc import Callable
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from app.services.llm.interfaces import (
    LLMAuthenticationError,
    LLMMalformedResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    SourceIdsShapeDiagnostic,
    LLMTimeoutError,
    LLMTransientError,
    inspect_source_ids_shape,
)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_MAX_RETRIES = 1
_MAX_RETRY_DELAY_SECONDS = 2.0
def _response_json_schema(
    required_source_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    source_ids_schema: dict[str, object] = {
        "type": "array",
        "items": {"type": "string"},
    }
    insufficient_evidence_schema: dict[str, object] = {"type": "boolean"}
    if required_source_ids is not None:
        source_ids_schema = {
            "type": "array",
            "items": {"type": "string"},
            "enum": [list(required_source_ids)],
        }
        insufficient_evidence_schema = {
            "type": "boolean",
            "enum": [False],
        }
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "source_ids": source_ids_schema,
            "insufficient_evidence": insufficient_evidence_schema,
        },
        "required": ["answer", "source_ids", "insufficient_evidence"],
        "additionalProperties": False,
    }


_RESPONSE_JSON_SCHEMA = _response_json_schema()

Sleep = Callable[[float], None]


def _malformed_response(
    message: str,
    *,
    diagnostic_category: str,
    actual_response_type: str,
    source_ids_rejected: bool = False,
    source_ids_shape: SourceIdsShapeDiagnostic | None = None,
) -> LLMMalformedResponseError:
    return LLMMalformedResponseError(
        message,
        diagnostic_category=diagnostic_category,
        actual_response_type=actual_response_type,
        source_ids_rejected=source_ids_rejected,
        source_ids_shape=source_ids_shape,
    )


class OllamaProvider:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        *,
        sleep: Sleep = time.sleep,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must not be empty.")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must not be empty.")
        parsed_url = urlsplit(base_url.strip())
        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
        ):
            raise ValueError("base_url must be an HTTP(S) URL without credentials.")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or timeout_seconds < 1
        ):
            raise ValueError("timeout_seconds must be a positive integer.")
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer.")

        self.model = model.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._sleep = sleep

    @staticmethod
    def _parse_response(
        response_body: bytes,
        expected_source_ids: set[str] | None = None,
    ) -> LLMResponse:
        try:
            response_payload = json.loads(response_body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _malformed_response(
                "The Ollama response was not valid JSON.",
                diagnostic_category="invalid_json",
                actual_response_type="bytes",
            ) from None

        if not isinstance(response_payload, dict):
            raise _malformed_response(
                "The Ollama response was not a valid object.",
                diagnostic_category="field_type",
                actual_response_type=type(response_payload).__name__,
            )
        message = response_payload.get("message")
        if not isinstance(message, dict):
            raise _malformed_response(
                "The Ollama response did not contain a message.",
                diagnostic_category="missing_fields",
                actual_response_type=type(response_payload).__name__,
            )
        output_text = message.get("content")
        if not isinstance(output_text, str) or not output_text.strip():
            raise _malformed_response(
                "The Ollama response did not contain structured output.",
                diagnostic_category="field_type",
                actual_response_type=type(output_text).__name__,
            )
        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError:
            raise _malformed_response(
                "The Ollama response content was not valid JSON.",
                diagnostic_category="invalid_json",
                actual_response_type=type(output_text).__name__,
            ) from None

        if not isinstance(payload, dict) or set(payload) != {
            "answer",
            "source_ids",
            "insufficient_evidence",
        }:
            raise _malformed_response(
                "The Ollama response did not match the required response fields.",
                diagnostic_category="missing_fields",
                actual_response_type=type(payload).__name__,
            )
        answer = payload["answer"]
        source_ids = payload["source_ids"]
        insufficient_evidence = payload["insufficient_evidence"]
        source_ids_shape = inspect_source_ids_shape(
            source_ids,
            expected_source_ids,
            normalization_ran=False,
        )
        if not isinstance(answer, str):
            raise _malformed_response(
                "The Ollama answer must be text.",
                diagnostic_category="field_type",
                actual_response_type=type(answer).__name__,
            )
        if (
            not isinstance(source_ids, list)
            or any(
                not isinstance(source_id, str) or not source_id
                for source_id in source_ids
            )
            or len(set(source_ids)) != len(source_ids)
        ):
            raise _malformed_response(
                "The Ollama response contains invalid source IDs.",
                diagnostic_category="source_ids",
                actual_response_type=type(source_ids).__name__,
                source_ids_rejected=True,
                source_ids_shape=source_ids_shape,
            )
        if not isinstance(insufficient_evidence, bool):
            raise _malformed_response(
                "The Ollama response evidence status must be a boolean.",
                diagnostic_category="field_type",
                actual_response_type=type(insufficient_evidence).__name__,
            )
        if insufficient_evidence:
            if source_ids:
                raise _malformed_response(
                    "An insufficient-evidence response cannot cite sources.",
                    diagnostic_category="source_ids",
                    actual_response_type=type(source_ids).__name__,
                    source_ids_rejected=True,
                    source_ids_shape=source_ids_shape,
                )
        elif not answer.strip():
            raise _malformed_response(
                "A supported answer must include text and at least one source ID.",
                diagnostic_category="empty_answer",
                actual_response_type=type(answer).__name__,
            )
        elif not source_ids:
            raise _malformed_response(
                "A supported answer must include text and at least one source ID.",
                diagnostic_category="source_ids",
                actual_response_type=type(source_ids).__name__,
                source_ids_rejected=True,
                source_ids_shape=source_ids_shape,
            )
        return LLMResponse(answer, tuple(source_ids), insufficient_evidence)

    @staticmethod
    def _is_timeout(error: Exception) -> bool:
        seen: set[int] = set()
        current: BaseException | None = error
        while isinstance(current, BaseException) and id(current) not in seen:
            if isinstance(current, (TimeoutError, socket.timeout)):
                return True
            seen.add(id(current))
            if isinstance(current, urllib.error.URLError):
                current = current.reason
            else:
                current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _error_kind(error: Exception) -> str:
        if OllamaProvider._is_timeout(error):
            return "timeout"
        if isinstance(error, urllib.error.HTTPError):
            if error.code in (401, 403):
                return "authentication"
            if error.code == 429:
                return "rate_limit"
            if error.code in (408, 425) or error.code >= 500:
                return "transient"
            return "provider"
        if isinstance(error, (urllib.error.URLError, OSError)):
            return "transient"
        return "provider"

    def generate(self, request: LLMRequest) -> LLMResponse:
        required_source_ids: tuple[str, ...] | None = None
        if request.require_all_source_ids:
            required_source_ids = tuple(
                source.source_id for source in request.sources
            )
            if (
                not required_source_ids
                or any(not source_id for source_id in required_source_ids)
                or len(set(required_source_ids)) != len(required_source_ids)
            ):
                raise LLMProviderError(
                    "Ollama request contains invalid source identifiers."
                )

        request_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "format": _response_json_schema(required_source_ids),
            "stream": False,
            "options": {"num_predict": request.max_output_tokens},
        }
        try:
            body = json.dumps(request_payload).encode("utf-8")
        except (TypeError, ValueError):
            raise LLMProviderError("Ollama request could not be prepared.") from None

        url = f"{self.base_url}/api/chat"
        for retry_index in range(self.max_retries + 1):
            http_request = Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(http_request, timeout=self.timeout_seconds) as response:
                    response_body = response.read()
            except (
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                OSError,
            ) as error:
                error_kind = self._error_kind(error)
                can_retry = error_kind in {"timeout", "rate_limit", "transient"}
                if can_retry and retry_index < self.max_retries:
                    delay = min(0.25 * (2**retry_index), _MAX_RETRY_DELAY_SECONDS)
                    self._sleep(delay)
                    continue
                message = "Ollama request failed."
                if error_kind == "timeout":
                    raise LLMTimeoutError(message) from None
                if error_kind == "rate_limit":
                    raise LLMRateLimitError(message) from None
                if error_kind == "authentication":
                    raise LLMAuthenticationError(message) from None
                if error_kind == "transient":
                    raise LLMTransientError(message) from None
                raise LLMProviderError(message) from None
            return self._parse_response(
                response_body,
                {source.source_id for source in request.sources},
            )

        raise LLMProviderError("Ollama request failed after configured retries.")
