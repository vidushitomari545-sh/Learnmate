"""OpenAI Responses API adapter for the provider-neutral LLM contract."""

import json
import os
import time
from collections.abc import Callable, Mapping
from typing import Protocol

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

DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_MAX_RETRIES = 1
_MAX_RETRY_DELAY_SECONDS = 2.0
_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
        "insufficient_evidence": {"type": "boolean"},
    },
    "required": ["answer", "source_ids", "insufficient_evidence"],
    "additionalProperties": False,
}


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


class _ResponsesResource(Protocol):
    def create(self, **kwargs: object) -> object: ...


class _OpenAIClient(Protocol):
    responses: _ResponsesResource


ClientFactory = Callable[..., _OpenAIClient]
Sleep = Callable[[float], None]


def _create_openai_client(**kwargs: object) -> _OpenAIClient:
    from openai import OpenAI

    return OpenAI(**kwargs)


class OpenAIProvider:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        *,
        api_key: str | None = None,
        client: _OpenAIClient | None = None,
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
            configured_key = api_key if api_key is not None else os.environ.get(
                "OPENAI_API_KEY"
            )
            if not configured_key or not configured_key.strip():
                raise ValueError(
                    "OPENAI_API_KEY is required when the OpenAI provider is selected."
                )
            factory = client_factory or _create_openai_client
            self._client = factory(
                api_key=configured_key,
                timeout=timeout_seconds,
                max_retries=0,
            )
        else:
            self._client = client

    @staticmethod
    def _error_kind(error: Exception) -> str:
        name = type(error).__name__.lower()
        status_code = getattr(error, "status_code", None)
        if "timeout" in name:
            return "timeout"
        if "authentication" in name or status_code in (401, 403):
            return "authentication"
        if "ratelimit" in name or status_code == 429:
            return "rate_limit"
        if (
            "connection" in name
            or "internalserver" in name
            or isinstance(status_code, int)
            and status_code >= 500
        ):
            return "transient"
        return "provider"

    @staticmethod
    def _retry_delay(error: Exception, retry_index: int) -> float:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        retry_after = headers.get("retry-after") if isinstance(headers, Mapping) else None
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            delay = 0.25 * (2**retry_index)
        return min(max(delay, 0.0), _MAX_RETRY_DELAY_SECONDS)

    @staticmethod
    def _parse_response(
        response: object,
        expected_source_ids: set[str] | None = None,
    ) -> LLMResponse:
        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise _malformed_response(
                "The OpenAI response did not contain structured output.",
                diagnostic_category="field_type",
                actual_response_type=type(output_text).__name__,
            )
        try:
            payload = json.loads(output_text)
        except json.JSONDecodeError:
            raise _malformed_response(
                "The OpenAI response was not valid JSON.",
                diagnostic_category="invalid_json",
                actual_response_type=type(output_text).__name__,
            ) from None

        if not isinstance(payload, dict) or set(payload) != {
            "answer",
            "source_ids",
            "insufficient_evidence",
        }:
            raise _malformed_response(
                "The OpenAI response did not match the required response fields.",
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
                "The OpenAI answer must be text.",
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
                "The OpenAI response contains invalid source IDs.",
                diagnostic_category="source_ids",
                actual_response_type=type(source_ids).__name__,
                source_ids_rejected=True,
                source_ids_shape=source_ids_shape,
            )
        if not isinstance(insufficient_evidence, bool):
            raise _malformed_response(
                "The OpenAI response evidence status must be a boolean.",
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

    def generate(self, request: LLMRequest) -> LLMResponse:
        request_payload: dict[str, object] = {
            "model": self.model,
            "instructions": request.system_prompt,
            "input": request.user_prompt,
            "max_output_tokens": request.max_output_tokens,
            "tools": [],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "learnmate_rag_response",
                    "strict": True,
                    "schema": _RESPONSE_SCHEMA,
                }
            },
        }

        for retry_index in range(self.max_retries + 1):
            try:
                raw_response = self._client.responses.create(**request_payload)
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
                    self._sleep(self._retry_delay(error, retry_index))
                    continue
                message = "OpenAI request failed."
                if error_kind == "timeout":
                    raise LLMTimeoutError(message) from error
                if error_kind == "rate_limit":
                    raise LLMRateLimitError(message) from error
                if error_kind == "authentication":
                    raise LLMAuthenticationError(message) from error
                if error_kind == "transient":
                    raise LLMTransientError(message) from error
                raise LLMProviderError(message) from error

        raise LLMProviderError("OpenAI request failed after configured retries.")
