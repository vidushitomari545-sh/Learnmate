import json
import socket
import ssl

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.interfaces import (
    LLMAuthenticationError,
    LLMMalformedResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMTransientError,
)


class APIResponse:
    def __init__(self, text: object) -> None:
        self.text = text


class MockModels:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, object]] = []

    def generate_content(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class MockClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.models = MockModels(outcomes)


def request() -> LLMRequest:
    return LLMRequest(
        question="What is the fact?",
        system_prompt="Use evidence only.",
        user_prompt='{"source_id":"S1","text":"Fact"}',
        sources=(),
        max_output_tokens=80,
    )


def response_json(
    answer: str = "The fact.",
    source_ids: list[str] | None = None,
    insufficient_evidence: bool = False,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "source_ids": ["S1"] if source_ids is None else source_ids,
            "insufficient_evidence": insufficient_evidence,
        }
    )


def test_sends_gemini_request_and_maps_response() -> None:
    client = MockClient([APIResponse(response_json())])
    provider = GeminiProvider(client=client, model="gemini-test-model")

    result = provider.generate(request())

    assert result == LLMResponse("The fact.", ("S1",), False)
    sent = client.models.requests[0]
    assert sent["model"] == "gemini-test-model"
    assert sent["contents"] == request().user_prompt
    generation_config = sent["config"]
    assert generation_config.system_instruction == "Use evidence only."
    assert generation_config.max_output_tokens == 80
    assert generation_config.response_mime_type == "application/json"
    assert generation_config.response_json_schema == {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "source_ids": {"type": "array", "items": {"type": "string"}},
            "insufficient_evidence": {"type": "boolean"},
        },
        "required": ["answer", "source_ids", "insufficient_evidence"],
        "additionalProperties": False,
    }
    assert generation_config.response_schema is None


def test_successful_response_parses_without_failure_stage() -> None:
    provider = GeminiProvider(client=MockClient([APIResponse(response_json())]))

    result = provider.generate(request())

    assert result == LLMResponse("The fact.", ("S1",), False)


def test_maps_insufficient_evidence_response() -> None:
    provider = GeminiProvider(
        client=MockClient(
            [APIResponse(response_json("Not supported.", [], insufficient_evidence=True))]
        )
    )

    assert provider.generate(request()) == LLMResponse("Not supported.", (), True)


def test_rejects_response_with_unreadable_text() -> None:
    class UnreadableResponse:
        @property
        def text(self) -> str:
            raise ValueError("provider response details")

    provider = GeminiProvider(client=MockClient([UnreadableResponse()]))

    with pytest.raises(LLMMalformedResponseError) as raised:
        provider.generate(request())

    assert "provider response details" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.failure_stage == "GEMINI_RESPONSE_EXTRACTION"


@pytest.mark.parametrize(
    "response",
    [
        APIResponse(None),
        types.GenerateContentResponse(),
        types.GenerateContentResponse(candidates=[types.Candidate(content=None)]),
        types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=None))]
        ),
    ],
)
def test_missing_candidate_content_parts_or_text_has_extraction_stage(
    response: object,
) -> None:
    provider = GeminiProvider(client=MockClient([response]))

    with pytest.raises(LLMMalformedResponseError) as raised:
        provider.generate(request())

    assert raised.value.failure_stage == "GEMINI_RESPONSE_EXTRACTION"


@pytest.mark.parametrize(
    ("payload", "failure_stage"),
    [
        ("synthetic malformed JSON", "GEMINI_JSON_DECODING"),
        ('{"answer": "ok"}', "GEMINI_SCHEMA_VALIDATION"),
        (
            '{"answer": 123, "source_ids": ["S1"], "insufficient_evidence": false}',
            "GEMINI_FIELD_TYPE_VALIDATION",
        ),
        (
            '{"answer": "ok", "source_ids": [1], "insufficient_evidence": false}',
            "GEMINI_FIELD_TYPE_VALIDATION",
        ),
        (
            '{"answer": "ok", "source_ids": ["S1"], "insufficient_evidence": 1}',
            "GEMINI_FIELD_TYPE_VALIDATION",
        ),
    ],
)
def test_assigns_distinct_json_failure_stages(
    payload: str,
    failure_stage: str,
) -> None:
    provider = GeminiProvider(client=MockClient([APIResponse(payload)]))

    with pytest.raises(LLMMalformedResponseError) as raised:
        provider.generate(request())

    assert raised.value.failure_stage == failure_stage
    assert "synthetic malformed JSON" not in str(raised.value)


def test_duplicate_source_ids_are_deferred_to_rag_validation() -> None:
    provider = GeminiProvider(
        client=MockClient(
            [
                APIResponse(
                    response_json("Answer.", ["S1", "S1"], insufficient_evidence=False)
                )
            ]
        )
    )

    assert provider.generate(request()) == LLMResponse(
        "Answer.", ("S1", "S1"), False
    )


def test_missing_api_key_fails_before_client_factory() -> None:
    constructed: list[bool] = []

    def client_factory(**kwargs: object) -> MockClient:
        constructed.append(True)
        return MockClient([])

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiProvider(api_key="", client_factory=client_factory)

    assert constructed == []


def test_client_factory_receives_key_timeout_and_no_sdk_retries() -> None:
    received: list[dict[str, object]] = []

    def client_factory(**kwargs: object) -> MockClient:
        received.append(kwargs)
        return MockClient([])

    GeminiProvider(
        api_key="test-only-secret",
        timeout_seconds=7,
        client_factory=client_factory,
    )

    assert len(received) == 1
    options = received[0]["http_options"]
    assert options.timeout == 7000
    assert options.retry_options is None
    assert received[0]["api_key"] == "test-only-secret"


@pytest.mark.parametrize(
    ("error", "expected_type", "status_code"),
    [
        (type("TimeoutError", (Exception,), {})("timed out"), LLMTimeoutError, None),
        (type("ClientError", (Exception,), {})("limited"), LLMRateLimitError, 429),
        (
            type("ClientError", (Exception,), {})("unauthorized"),
            LLMAuthenticationError,
            401,
        ),
        (
            type("ServerError", (Exception,), {})("server down"),
            LLMTransientError,
            503,
        ),
        (
            type("ClientError", (Exception,), {})("bad request"),
            LLMProviderError,
            400,
        ),
    ],
)
def test_maps_provider_errors_without_exposing_provider_details(
    error: Exception,
    expected_type: type[Exception],
    status_code: int | None,
) -> None:
    if status_code is not None:
        error.code = status_code  # type: ignore[attr-defined]
    provider = GeminiProvider(client=MockClient([error]), max_retries=0)

    with pytest.raises(expected_type) as raised:
        provider.generate(request())

    assert str(raised.value) == "Gemini request failed."
    assert "server down" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    ("error", "mapped_type", "exception_category"),
    [
        (httpx.ConnectError("private connection detail"), LLMTransientError, "connection"),
        (httpx.ConnectTimeout("private timeout detail"), LLMTimeoutError, "timeout"),
    ],
)
def test_classifies_transport_errors_without_exposing_messages(
    error: Exception,
    mapped_type: type[Exception],
    exception_category: str,
) -> None:
    client = MockClient([error])
    provider = GeminiProvider(client=client, max_retries=0)

    with pytest.raises(mapped_type) as raised:
        provider.generate(request())

    assert len(client.models.requests) == 1
    assert raised.value.provider_exception_class == type(error).__name__
    assert raised.value.provider_exception_category == exception_category
    assert raised.value.error_category in {"transient", "timeout"}
    assert raised.value.attempt_count == 1
    assert "private" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    ("underlying_error", "exception_category"),
    [
        (socket.gaierror("private DNS detail"), "dns"),
        (ssl.SSLError("private TLS detail"), "tls"),
    ],
)
def test_classifies_dns_and_tls_connect_causes_safely(
    underlying_error: Exception,
    exception_category: str,
) -> None:
    connection_error = httpx.ConnectError("private connection detail")
    connection_error.__cause__ = underlying_error
    client = MockClient([connection_error])
    provider = GeminiProvider(client=client, max_retries=0)

    with pytest.raises(LLMTransientError) as raised:
        provider.generate(request())

    assert len(client.models.requests) == 1
    assert raised.value.provider_exception_class == "ConnectError"
    assert raised.value.provider_exception_category == exception_category
    assert raised.value.error_category == "transient"
    assert raised.value.attempt_count == 1
    assert "private" not in str(raised.value)


def test_connection_error_uses_bounded_adapter_retry() -> None:
    client = MockClient(
        [
            httpx.ConnectError("private connection detail"),
            APIResponse(response_json()),
        ]
    )
    provider = GeminiProvider(client=client, max_retries=1, sleep=lambda _: None)

    assert provider.generate(request()) == LLMResponse("The fact.", ("S1",), False)
    assert len(client.models.requests) == 2


def test_unknown_exception_has_safe_metadata_without_retry() -> None:
    client = MockClient([RuntimeError("secret API key and headers")])
    provider = GeminiProvider(client=client, max_retries=2, sleep=lambda _: None)

    with pytest.raises(LLMProviderError) as raised:
        provider.generate(request())

    assert len(client.models.requests) == 1
    assert raised.value.provider_exception_class == "RuntimeError"
    assert raised.value.provider_exception_category == "unknown"
    assert raised.value.error_category == "provider"
    assert raised.value.attempt_count == 1
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    ("status_code", "api_status", "expected_type", "category"),
    [
        (429, "RESOURCE_EXHAUSTED", LLMRateLimitError, "rate_limit"),
        (503, "UNAVAILABLE", LLMTransientError, "transient"),
        (500, "INTERNAL", LLMTransientError, "transient"),
    ],
)
def test_preserves_safe_metadata_for_known_api_errors(
    status_code: int,
    api_status: str,
    expected_type: type[Exception],
    category: str,
) -> None:
    error = genai_errors.ClientError(
        status_code,
        {"error": {"status": api_status, "message": "sensitive provider text"}},
    )
    client = MockClient([error])
    provider = GeminiProvider(client=client, max_retries=0)

    with pytest.raises(expected_type) as raised:
        provider.generate(request())

    assert len(client.models.requests) == 1
    assert str(raised.value) == "Gemini request failed."
    assert raised.value.provider_exception_class == "ClientError"
    assert raised.value.http_status_code == status_code
    assert raised.value.provider_status == api_status
    assert raised.value.error_category == category
    assert raised.value.provider_exception_category == f"api_{category}"
    assert raised.value.attempt_count == 1
    assert "sensitive provider text" not in repr(vars(raised.value))
    assert raised.value.__cause__ is None


def test_unknown_api_error_preserves_only_allowlisted_metadata() -> None:
    error = genai_errors.ClientError(
        400,
        {
            "error": {
                "status": "PRIVATE_PROVIDER_STATUS",
                "message": "sensitive response text",
                "details": [{"api_key": "test-secret", "headers": "private"}],
            }
        },
    )
    client = MockClient([error])
    provider = GeminiProvider(client=client, max_retries=0)

    with pytest.raises(LLMProviderError) as raised:
        provider.generate(request())

    assert len(client.models.requests) == 1
    assert str(raised.value) == "Gemini request failed."
    assert raised.value.provider_exception_class == "ClientError"
    assert raised.value.http_status_code == 400
    assert raised.value.provider_status is None
    assert raised.value.error_category == "provider"
    assert raised.value.provider_exception_category == "api_other"
    assert raised.value.attempt_count == 1
    assert "sensitive response text" not in repr(vars(raised.value))
    assert "test-secret" not in repr(vars(raised.value))
    assert "private" not in repr(vars(raised.value))


def test_api_error_metadata_reports_total_adapter_attempts() -> None:
    errors = [
        genai_errors.ServerError(
            503,
            {"error": {"status": "UNAVAILABLE", "message": "temporary"}},
        )
        for _ in range(3)
    ]
    client = MockClient(errors)
    provider = GeminiProvider(client=client, max_retries=2, sleep=lambda _: None)

    with pytest.raises(LLMTransientError) as raised:
        provider.generate(request())

    assert len(client.models.requests) == 3
    assert raised.value.attempt_count == 3
    assert raised.value.http_status_code == 503
    assert raised.value.provider_status == "UNAVAILABLE"


def test_retries_transient_failures_only_within_configured_bound() -> None:
    failure = type("ServerError", (Exception,), {})("server down")
    failure.code = 503  # type: ignore[attr-defined]
    client = MockClient([failure, APIResponse(response_json())])
    slept: list[float] = []
    provider = GeminiProvider(client=client, max_retries=1, sleep=slept.append)

    result = provider.generate(request())

    assert result.answer == "The fact."
    assert len(client.models.requests) == 2
    assert slept == [pytest.approx(0.25)]


def test_retryable_errors_stop_at_initial_attempt_plus_retry_limit() -> None:
    errors = []
    for _ in range(3):
        error = type("ServerError", (Exception,), {})("server down")
        error.code = 503  # type: ignore[attr-defined]
        errors.append(error)
    client = MockClient(errors)
    provider = GeminiProvider(client=client, max_retries=2, sleep=lambda _: None)

    with pytest.raises(LLMTransientError):
        provider.generate(request())

    assert len(client.models.requests) == 3


@pytest.mark.parametrize(
    "error",
    [
        type("AuthenticationError", (Exception,), {})("unauthorized"),
        type("ClientError", (Exception,), {})("invalid request"),
    ],
)
def test_non_retryable_errors_use_exactly_one_attempt(error: Exception) -> None:
    error.code = 401 if type(error).__name__ == "AuthenticationError" else 400  # type: ignore[attr-defined]
    client = MockClient([error])
    provider = GeminiProvider(client=client, max_retries=3)

    with pytest.raises((LLMAuthenticationError, LLMProviderError)):
        provider.generate(request())

    assert len(client.models.requests) == 1


def test_secret_and_headers_are_not_exposed_in_wrapped_error() -> None:
    secret = "test-secret-value"
    error = type("AuthenticationError", (Exception,), {})(
        f"unauthorized, api_key={secret}, headers={{Authorization: Bearer {secret}}}"
    )
    error.code = 401  # type: ignore[attr-defined]
    provider = GeminiProvider(client=MockClient([error]), max_retries=0)

    with pytest.raises(LLMAuthenticationError) as raised:
        provider.generate(request())

    assert secret not in str(raised.value)
    assert "Authorization" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    ("model", "timeout_seconds", "max_retries"),
    [
        (" ", 20, 1),
        ("gemini-test-model", 0, 1),
        ("gemini-test-model", True, 1),
        ("gemini-test-model", 20, -1),
        ("gemini-test-model", 20, True),
    ],
)
def test_rejects_invalid_provider_configuration(
    model: str, timeout_seconds: int, max_retries: int
) -> None:
    with pytest.raises(ValueError):
        GeminiProvider(
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            client=MockClient([]),
        )
