import json
import socket
import urllib.error
from urllib.request import Request

import pytest

from app.services.llm.interfaces import (
    LLMAuthenticationError,
    LLMContextSource,
    LLMMalformedResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMTransientError,
)
from app.services.llm.ollama_provider import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    OllamaProvider,
)


class MockResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self) -> "MockResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None


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
) -> bytes:
    content = json.dumps(
        {
            "answer": answer,
            "source_ids": ["S1"] if source_ids is None else source_ids,
            "insufficient_evidence": insufficient_evidence,
        }
    )
    return json.dumps({"message": {"role": "assistant", "content": content}}).encode()


def test_sends_chat_request_and_maps_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[Request, int]] = []

    def fake_urlopen(http_request: Request, timeout: int) -> MockResponse:
        sent.append((http_request, timeout))
        return MockResponse(response_json())

    monkeypatch.setattr("app.services.llm.ollama_provider.urlopen", fake_urlopen)
    provider = OllamaProvider(model="llama3.2:3b", timeout_seconds=9)

    result = provider.generate(request())

    assert result == LLMResponse("The fact.", ("S1",), False)
    http_request, timeout = sent[0]
    assert http_request.full_url == f"{DEFAULT_BASE_URL}/api/chat"
    assert http_request.get_method() == "POST"
    assert http_request.get_header("Content-type") == "application/json"
    assert timeout == 9
    payload = json.loads(http_request.data)
    assert payload["model"] == DEFAULT_MODEL
    assert payload["messages"] == [
        {"role": "system", "content": "Use evidence only."},
        {"role": "user", "content": request().user_prompt},
    ]
    assert payload["format"] == {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "source_ids": {"type": "array", "items": {"type": "string"}},
            "insufficient_evidence": {"type": "boolean"},
        },
        "required": ["answer", "source_ids", "insufficient_evidence"],
        "additionalProperties": False,
    }
    assert payload["stream"] is False
    assert payload["options"] == {"num_predict": 80}


def test_summary_request_constrains_response_to_all_supplied_source_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[Request] = []

    def fake_urlopen(http_request: Request, timeout: int) -> MockResponse:
        sent.append(http_request)
        return MockResponse(
            response_json(source_ids=["C000001", "C000002"])
        )

    monkeypatch.setattr("app.services.llm.ollama_provider.urlopen", fake_urlopen)
    summary_request = LLMRequest(
        question="Summarize the supplied PDF section.",
        system_prompt="Summarize only the supplied sources.",
        user_prompt="Sources C000001 and C000002.",
        sources=(
            LLMContextSource("C000001", "First source."),
            LLMContextSource("C000002", "Second source."),
        ),
        max_output_tokens=80,
        require_all_source_ids=True,
    )

    result = OllamaProvider().generate(summary_request)

    assert result.source_ids == ("C000001", "C000002")
    schema = json.loads(sent[0].data)["format"]
    assert schema["properties"]["source_ids"] == {
        "type": "array",
        "items": {"type": "string"},
        "enum": [["C000001", "C000002"]],
    }
    assert schema["properties"]["insufficient_evidence"] == {
        "type": "boolean",
        "enum": [False],
    }


def test_maps_insufficient_evidence_response() -> None:
    assert OllamaProvider._parse_response(
        response_json("Not supported.", [], insufficient_evidence=True)
    ) == LLMResponse("Not supported.", (), True)


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        json.dumps({"message": {"content": "not-json"}}).encode(),
        json.dumps({"message": {"content": "{}"}}).encode(),
        json.dumps(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "answer": "ok",
                            "source_ids": ["S1", "S1"],
                            "insufficient_evidence": False,
                        }
                    )
                }
            }
        ).encode(),
        response_json("Not supported.", ["S1"], insufficient_evidence=True),
        response_json("", [], insufficient_evidence=False),
    ],
)
def test_rejects_malformed_structured_responses(body: bytes) -> None:
    with pytest.raises(LLMMalformedResponseError):
        OllamaProvider._parse_response(body)


@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (
            urllib.error.URLError(ConnectionRefusedError("private connection detail")),
            LLMTransientError,
        ),
        (socket.timeout("private timeout detail"), LLMTimeoutError),
        (
            urllib.error.HTTPError("http://localhost", 429, "limited", None, None),
            LLMRateLimitError,
        ),
        (
            urllib.error.HTTPError("http://localhost", 401, "unauthorized", None, None),
            LLMAuthenticationError,
        ),
        (
            urllib.error.HTTPError("http://localhost", 400, "bad request", None, None),
            LLMProviderError,
        ),
    ],
)
def test_maps_request_errors_without_exposing_provider_details(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_type: type[Exception],
) -> None:
    def fail_urlopen(_: Request, timeout: int) -> MockResponse:
        raise error

    monkeypatch.setattr("app.services.llm.ollama_provider.urlopen", fail_urlopen)
    provider = OllamaProvider(max_retries=0)

    with pytest.raises(expected_type) as raised:
        provider.generate(request())

    assert str(raised.value) == "Ollama request failed."
    assert "private" not in str(raised.value)


def test_retries_connection_errors_within_configured_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = [
        urllib.error.URLError(ConnectionRefusedError("connection refused")),
        MockResponse(response_json()),
    ]
    attempts: list[bool] = []
    slept: list[float] = []

    def fake_urlopen(_: Request, timeout: int) -> MockResponse:
        attempts.append(True)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("app.services.llm.ollama_provider.urlopen", fake_urlopen)
    provider = OllamaProvider(max_retries=1, sleep=slept.append)

    assert provider.generate(request()).answer == "The fact."
    assert len(attempts) == 2
    assert slept == [pytest.approx(0.25)]
