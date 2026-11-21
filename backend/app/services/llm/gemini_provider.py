"""Google Gemini adapter for the provider-neutral LLM contract."""

import json
import os
import socket
import ssl
import time
from collections.abc import Callable
from typing import Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app import config
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

DEFAULT_MODEL = config.GEMINI_MODEL
DEFAULT_TIMEOUT_SECONDS = config.LLM_TIMEOUT_SECONDS
DEFAULT_MAX_RETRIES = config.LLM_MAX_RETRIES
_MAX_RETRY_DELAY_SECONDS = 2.0
_SAFE_API_STATUSES = {
    "DEADLINE_EXCEEDED",
    "INTERNAL",
    "INVALID_ARGUMENT",
    "PERMISSION_DENIED",
    "RESOURCE_EXHAUSTED",
    "UNAUTHENTICATED",
    "UNAVAILABLE",
    "UNKNOWN",
}
_SAFE_EXCEPTION_CATEGORIES = {
    "api_authentication",
    "api_other",
    "api_rate_limit",
    "api_transient",
    "connection",
    "dns",
    "timeout",
    "tls",
    "unknown",
}
_RESPONSE_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
        "insufficient_evidence": {"type": "boolean"},
    },
    "required": ["answer", "source_ids", "insufficient_evidence"],
    "additionalProperties": False,
}
_GEMINI_RESPONSE_EXTRACTION = "GEMINI_RESPONSE_EXTRACTION"
_GEMINI_JSON_PARSING = "GEMINI_JSON_PARSING"
_GEMINI_JSON_DECODING = "GEMINI_JSON_DECODING"
_GEMINI_SCHEMA_VALIDATION = "GEMINI_SCHEMA_VALIDATION"
_GEMINI_FIELD_TYPE_VALIDATION = "GEMINI_FIELD_TYPE_VALIDATION"


def _malformed_response(
    message: str,
    failure_stage: str,
    *,
    diagnostic_category: str,
    actual_response_type: str,
    source_ids_rejected: bool = False,
    source_ids_shape: SourceIdsShapeDiagnostic | None = None,
) -> LLMMalformedResponseError:
    error = LLMMalformedResponseError(
        message,
        diagnostic_category=diagnostic_category,
        actual_response_type=actual_response_type,
        source_ids_rejected=source_ids_rejected,
        source_ids_shape=source_ids_shape,
    )
    error.failure_stage = failure_stage
    return error


class _ModelsResource(Protocol):
    def generate_content(self, **kwargs: object) -> object: ...


class _GeminiClient(Protocol):
    models: _ModelsResource


ClientFactory = Callable[..., _GeminiClient]
Sleep = Callable[[float], None]


def _create_gemini_client(**kwargs: object) -> _GeminiClient:
    return genai.Client(**kwargs)


class GeminiProvider:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        *,
        api_key: str | None = None,
        client: _GeminiClient | None = None,
        client_factory: ClientFactory | None = None,
        sleep: Sleep = time.sleep,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must not be empty.")
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

        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._sleep = sleep
        if client is None:
            configured_key = (
                api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
            )
            if not configured_key or not configured_key.strip():
                raise ValueError(
                    "GEMINI_API_KEY is required when the Gemini provider is selected."
                )
            factory = client_factory or _create_gemini_client
            try:
                self._client = factory(
                    api_key=configured_key,
                    http_options=types.HttpOptions(
                        timeout=timeout_seconds * 1000,
                        retry_options=None,
                    ),
                )
            except Exception:
                raise ValueError("Gemini client initialization failed.") from None
        else:
            self._client = client

    @staticmethod
    def _exception_chain(error: Exception) -> tuple[Exception, ...]:
        chain: list[Exception] = []
        seen: set[int] = set()
        current: BaseException | None = error
        while isinstance(current, Exception) and id(current) not in seen:
            chain.append(current)
            seen.add(id(current))
            current = current.__cause__ or current.__context__
        return tuple(chain)

    @classmethod
    def _exception_category(cls, error: Exception, error_kind: str) -> str:
        if isinstance(error, genai_errors.APIError):
            return {
                "authentication": "api_authentication",
                "rate_limit": "api_rate_limit",
                "transient": "api_transient",
            }.get(error_kind, "api_other")

        chain = cls._exception_chain(error)
        if any(isinstance(item, ssl.SSLError) for item in chain):
            return "tls"
        if any(isinstance(item, socket.gaierror) for item in chain):
            return "dns"

        for item in chain:
            name = type(item).__name__.lower()
            module = type(item).__module__
            if "timeout" in name:
                return "timeout"
            if (
                module.startswith(("httpx", "httpcore"))
                and name
                in {
                    "connecterror",
                    "networkerror",
                    "proxyerror",
                    "remoteprotocolerror",
                }
            ):
                return "connection"
        return "unknown"

    @classmethod
    def _error_kind(cls, error: Exception) -> str:
        name = type(error).__name__.lower()
        status_code = getattr(error, "code", getattr(error, "status_code", None))
        api_status = getattr(error, "status", None)
        if isinstance(api_status, str):
            api_status = api_status.upper()
        else:
            api_status = None
        exception_category = cls._exception_category(error, "provider")
        if exception_category == "timeout" or "timeout" in name:
            return "timeout"
        if (
            "authentication" in name
            or status_code in (401, 403)
            or api_status in {"UNAUTHENTICATED", "PERMISSION_DENIED"}
        ):
            return "authentication"
        if (
            "ratelimit" in name
            or status_code == 429
            or api_status == "RESOURCE_EXHAUSTED"
        ):
            return "rate_limit"
        if (
            "connection" in name
            or api_status in {"UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED"}
            or exception_category in {"connection", "dns", "tls"}
            or isinstance(status_code, int)
            and status_code in (408, 500, 502, 503, 504)
            or isinstance(status_code, int)
            and status_code >= 500
        ):
            return "transient"
        return "provider"

    @classmethod
    def _attach_error_metadata(
        cls,
        target: LLMProviderError,
        error: Exception,
        error_kind: str,
        attempt_count: int,
    ) -> None:
        exception_class = type(error).__name__
        if (
            not exception_class.isascii()
            or not exception_class.replace("_", "").isalnum()
            or len(exception_class) > 80
        ):
            exception_class = "UnknownException"

        status_code = getattr(error, "code", getattr(error, "status_code", None))
        if isinstance(status_code, bool) or not isinstance(status_code, int):
            status_code = None

        api_status = getattr(error, "status", None)
        if isinstance(api_status, str):
            api_status = api_status.upper()
        if api_status not in _SAFE_API_STATUSES:
            api_status = None

        target.provider_exception_class = exception_class
        target.http_status_code = status_code
        target.provider_status = api_status
        target.error_category = error_kind
        exception_category = cls._exception_category(error, error_kind)
        target.provider_exception_category = (
            exception_category
            if exception_category in _SAFE_EXCEPTION_CATEGORIES
            else "unknown"
        )
        target.attempt_count = attempt_count

    @staticmethod
    def _parse_response(
        response: object,
        expected_source_ids: set[str] | None = None,
    ) -> LLMResponse:
        try:
            output_text = getattr(response, "text", None)
        except Exception:
            raise _malformed_response(
                "The Gemini response did not contain readable structured output.",
                _GEMINI_RESPONSE_EXTRACTION,
                diagnostic_category="field_type",
                actual_response_type="unavailable",
            ) from None
        if not isinstance(output_text, str) or not output_text.strip():
            raise _malformed_response(
                "The Gemini response did not contain structured output.",
                _GEMINI_RESPONSE_EXTRACTION,
                diagnostic_category="field_type",
                actual_response_type=type(output_text).__name__,
            )
        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError:
            raise _malformed_response(
                "The Gemini response was not valid JSON.",
                _GEMINI_JSON_DECODING,
                diagnostic_category="invalid_json",
                actual_response_type=type(output_text).__name__,
            ) from None

        if not isinstance(payload, dict) or set(payload) != {
            "answer",
            "source_ids",
            "insufficient_evidence",
        }:
            raise _malformed_response(
                "The Gemini response did not match the required response fields.",
                _GEMINI_SCHEMA_VALIDATION,
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
                "The Gemini answer must be text.",
                _GEMINI_FIELD_TYPE_VALIDATION,
                diagnostic_category="field_type",
                actual_response_type=type(answer).__name__,
            )
        if (
            not isinstance(source_ids, list)
            or any(not isinstance(source_id, str) for source_id in source_ids)
        ):
            raise _malformed_response(
                "The Gemini response contains invalid source IDs.",
                _GEMINI_FIELD_TYPE_VALIDATION,
                diagnostic_category="source_ids",
                actual_response_type=type(source_ids).__name__,
                source_ids_rejected=True,
                source_ids_shape=source_ids_shape,
            )
        if not isinstance(insufficient_evidence, bool):
            raise _malformed_response(
                "The Gemini response evidence status must be a boolean.",
                _GEMINI_FIELD_TYPE_VALIDATION,
                diagnostic_category="field_type",
                actual_response_type=type(insufficient_evidence).__name__,
            )
        return LLMResponse(answer, tuple(source_ids), insufficient_evidence)

    def generate(self, request: LLMRequest) -> LLMResponse:
        request_config = types.GenerateContentConfig(
            system_instruction=request.system_prompt,
            max_output_tokens=request.max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=_RESPONSE_JSON_SCHEMA,
        )

        for retry_index in range(self.max_retries + 1):
            try:
                raw_response = self._client.models.generate_content(
                    model=self.model,
                    contents=request.user_prompt,
                    config=request_config,
                )
                return self._parse_response(
                    raw_response,
                    {source.source_id for source in request.sources},
                )
            except LLMMalformedResponseError:
                raise
            except Exception as error:
                error_kind = self._error_kind(error)
                can_retry = error_kind in {"timeout", "rate_limit", "transient"}
                if can_retry and retry_index < self.max_retries:
                    delay = min(0.25 * (2**retry_index), _MAX_RETRY_DELAY_SECONDS)
                    self._sleep(delay)
                    continue
                message = "Gemini request failed."
                attempt_count = retry_index + 1
                if error_kind == "timeout":
                    provider_error: LLMProviderError = LLMTimeoutError(message)
                elif error_kind == "rate_limit":
                    provider_error = LLMRateLimitError(message)
                elif error_kind == "authentication":
                    provider_error = LLMAuthenticationError(message)
                elif error_kind == "transient":
                    provider_error = LLMTransientError(message)
                else:
                    provider_error = LLMProviderError(message)
                self._attach_error_metadata(
                    provider_error,
                    error,
                    error_kind,
                    attempt_count,
                )
                raise provider_error from None

        raise LLMProviderError("Gemini request failed after configured retries.")
