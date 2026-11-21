import json

import pytest

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
from app.services.llm.openai_provider import OpenAIProvider


class APIResponse:
    def __init__(self, output_text: object) -> None:
        self.output_text = output_text


class MockResponses:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class MockClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.responses = MockResponses(outcomes)


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


def test_sends_responses_request_with_strict_structured_output() -> None:
    client = MockClient([APIResponse(response_json())])
    provider = OpenAIProvider(client=client, model="gpt-6-luna")

    result = provider.generate(request())

    assert result == LLMResponse("The fact.", ("S1",), False)
    sent = client.responses.requests[0]
    assert sent["model"] == "gpt-6-luna"
    assert sent["instructions"] == "Use evidence only."
    assert sent["input"] == '{"source_id":"S1","text":"Fact"}'
    assert sent["max_output_tokens"] == 80
    assert sent["tools"] == []
    assert sent["text"] == {
        "format": {
            "type": "json_schema",
            "name": "learnmate_rag_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                    "insufficient_evidence": {"type": "boolean"},
                },
                "required": ["answer", "source_ids", "insufficient_evidence"],
                "additionalProperties": False,
            },
        }
    }


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        "{}",
        '{"answer": 123, "source_ids": ["S1"], "insufficient_evidence": false}',
        '{"answer": "ok", "source_ids": ["S1", "S1"], "insufficient_evidence": false}',
        '{"answer": "ok", "source_ids": [], "insufficient_evidence": false}',
        '{"answer": "no", "source_ids": ["S1"], "insufficient_evidence": true}',
        '{"answer": "ok", "source_ids": ["S1"], "insufficient_evidence": 1}',
        '{"answer": "ok", "source_ids": ["S1"], "insufficient_evidence": false, "extra": 1}',
    ],
)
def test_rejects_malformed_structured_responses(payload: str) -> None:
    provider = OpenAIProvider(client=MockClient([APIResponse(payload)]))

    with pytest.raises(LLMMalformedResponseError):
        provider.generate(request())


def test_rejects_empty_or_refusal_only_response_as_malformed() -> None:
    provider = OpenAIProvider(client=MockClient([APIResponse("")]))

    with pytest.raises(LLMMalformedResponseError, match="structured output"):
        provider.generate(request())


def test_missing_api_key_fails_before_client_factory_or_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    constructed: list[bool] = []

    def client_factory(**kwargs: object) -> MockClient:
        constructed.append(True)
        return MockClient([])

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIProvider(client_factory=client_factory)

    assert constructed == []


def test_client_factory_receives_server_key_timeout_and_disabled_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    received: list[dict[str, object]] = []

    def client_factory(**kwargs: object) -> MockClient:
        received.append(kwargs)
        return MockClient([APIResponse(response_json())])

    provider = OpenAIProvider(
        model="custom-model",
        timeout_seconds=7,
        max_retries=0,
        client_factory=client_factory,
    )

    provider.generate(request())

    assert received == [
        {
            "api_key": "test-only-secret",
            "timeout": 7,
            "max_retries": 0,
        }
    ]


@pytest.mark.parametrize(
    ("error", "expected_type", "status_code"),
    [
        (type("APITimeoutError", (Exception,), {})("timeout"), LLMTimeoutError, None),
        (type("RateLimitError", (Exception,), {})("limited"), LLMRateLimitError, 429),
        (
            type("AuthenticationError", (Exception,), {})("bad credentials"),
            LLMAuthenticationError,
            401,
        ),
        (
            type("InternalServerError", (Exception,), {})("server down"),
            LLMTransientError,
            503,
        ),
        (
            type("APIStatusError", (Exception,), {})("bad request"),
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
        error.status_code = status_code  # type: ignore[attr-defined]
    provider = OpenAIProvider(client=MockClient([error]), max_retries=0)

    with pytest.raises(expected_type) as raised:
        provider.generate(request())

    assert str(raised.value) == "OpenAI request failed."
    assert "bad credentials" not in str(raised.value)


@pytest.mark.parametrize(
    ("first_error", "expected_type"),
    [
        (type("APITimeoutError", (Exception,), {})("timeout"), LLMTimeoutError),
        (type("RateLimitError", (Exception,), {})("limited"), LLMRateLimitError),
        (
            type("InternalServerError", (Exception,), {})("server down"),
            LLMTransientError,
        ),
    ],
)
def test_retries_transient_failures_only_within_configured_bound(
    first_error: Exception,
    expected_type: type[Exception],
) -> None:
    client = MockClient([first_error, APIResponse(response_json())])
    slept: list[float] = []
    provider = OpenAIProvider(
        client=client,
        max_retries=1,
        sleep=slept.append,
    )

    result = provider.generate(request())

    assert result.answer == "The fact."
    assert len(client.responses.requests) == 2
    assert slept == [pytest.approx(0.25)]

    failed_client = MockClient([first_error, first_error])
    failed_sleeps: list[float] = []
    exhausted_provider = OpenAIProvider(
        client=failed_client,
        max_retries=1,
        sleep=failed_sleeps.append,
    )
    with pytest.raises(expected_type):
        exhausted_provider.generate(request())
    assert len(failed_client.responses.requests) == 2
    assert failed_sleeps == [pytest.approx(0.25)]


def test_does_not_retry_authentication_errors() -> None:
    error = type("AuthenticationError", (Exception,), {})("unauthorized")
    error.status_code = 401  # type: ignore[attr-defined]
    client = MockClient([error])
    provider = OpenAIProvider(client=client, max_retries=3)

    with pytest.raises(LLMAuthenticationError):
        provider.generate(request())

    assert len(client.responses.requests) == 1


@pytest.mark.parametrize(
    ("model", "timeout_seconds", "max_retries"),
    [
        (" ", 20, 1),
        ("gpt-6-luna", 0, 1),
        ("gpt-6-luna", True, 1),
        ("gpt-6-luna", 20, -1),
        ("gpt-6-luna", 20, True),
    ],
)
def test_rejects_invalid_provider_configuration(
    model: str, timeout_seconds: int, max_retries: int
) -> None:
    with pytest.raises(ValueError):
        OpenAIProvider(
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            client=MockClient([]),
        )
