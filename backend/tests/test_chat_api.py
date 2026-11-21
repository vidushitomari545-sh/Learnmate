from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app import config
from app.api.chat import get_development_owner_id, get_rag_service
from app.main import app
from app.services.dependencies import RAGServiceFactory
from app.services.llm.fake import FakeLLMProvider
from app.services.llm.interfaces import (
    LLMAuthenticationError,
    LLMMalformedResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTransientError,
)
from app.services.rag_service import RAGCitation, RAGResult


class FakeRAGService:
    def __init__(self, result: RAGResult | None = None) -> None:
        self.result = result or RAGResult(
            "Grounded answer.", (), True
        )
        self.calls: list[tuple[str, str]] = []
        self.error: Exception | None = None

    def answer(self, owner_id: str, question: str) -> RAGResult:
        self.calls.append((owner_id, question))
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def client() -> Iterator[TestClient]:
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.pop(get_development_owner_id, None)
    app.dependency_overrides.pop(get_rag_service, None)


def override_dependencies(
    owner_id: str = "trusted-owner",
    service: FakeRAGService | None = None,
) -> FakeRAGService:
    rag_service = service or FakeRAGService()
    app.dependency_overrides[get_development_owner_id] = lambda: owner_id
    app.dependency_overrides[get_rag_service] = lambda: rag_service
    return rag_service


def test_chat_returns_structured_answer_and_trusted_citations(
    client: TestClient,
) -> None:
    service = FakeRAGService(
        RAGResult(
            "Photosynthesis stores energy in glucose.",
            (
                RAGCitation(
                    source_id="S1",
                    document_id="doc-1",
                    document_name="biology.pdf",
                    page_numbers=(4, 5),
                ),
            ),
            False,
        )
    )
    override_dependencies(service=service)

    response = client.post(
        "/api/chat",
        json={"question": "  What does photosynthesis store?  "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Photosynthesis stores energy in glucose.",
        "insufficient_evidence": False,
        "citations": [
            {"document_name": "biology.pdf", "page_numbers": [4, 5]}
        ],
    }
    assert service.calls == [
        ("trusted-owner", "What does photosynthesis store?")
    ]


def test_extra_client_owner_id_cannot_override_trusted_owner(
    client: TestClient,
) -> None:
    service = override_dependencies(owner_id="server-owner")

    response = client.post(
        "/api/chat",
        json={"question": "What is in my notes?", "owner_id": "attacker-owner"},
    )

    assert response.status_code == 422
    assert service.calls == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": " \t\n "},
        {"question": "x" * 4001},
        {"question": None},
    ],
)
def test_invalid_missing_or_oversized_questions_return_422(
    client: TestClient,
    payload: dict[str, object],
) -> None:
    service = override_dependencies()

    response = client.post("/api/chat", json=payload)

    assert response.status_code == 422
    assert service.calls == []


def test_empty_retrieval_returns_success_without_citations(
    client: TestClient,
) -> None:
    override_dependencies(
        service=FakeRAGService(
            RAGResult(
                "I couldn't find relevant information in your uploaded materials.",
                (),
                True,
            )
        )
    )

    response = client.post("/api/chat", json={"question": "Unanswerable question"})

    assert response.status_code == 200
    assert response.json()["insufficient_evidence"] is True
    assert response.json()["citations"] == []


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (
            LLMTimeoutError("secret timeout detail"),
            504,
            "The answer service timed out. Please try again.",
        ),
        (
            LLMRateLimitError("private rate detail"),
            429,
            "The answer service is temporarily rate limited. Please try later.",
        ),
        (
            LLMAuthenticationError("secret credential detail"),
            503,
            "The answer service is temporarily unavailable.",
        ),
        (
            LLMTransientError("private server detail"),
            503,
            "The answer service is temporarily unavailable.",
        ),
        (
            LLMMalformedResponseError("private malformed data"),
            502,
            "The answer service returned an invalid response.",
        ),
        (
            LLMProviderError("provider internal detail"),
            503,
            "The answer service is temporarily unavailable.",
        ),
        (
            RuntimeError("secret traceback detail"),
            500,
            "The question could not be processed.",
        ),
    ],
)
def test_maps_provider_and_internal_errors_to_safe_responses(
    client: TestClient,
    error: Exception,
    expected_status: int,
    expected_detail: str,
) -> None:
    service = FakeRAGService()
    service.error = error
    override_dependencies(service=service)

    response = client.post("/api/chat", json={"question": "Question"})

    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
    assert str(error) not in response.text


def test_provider_failure_logs_safe_metadata_without_sensitive_data(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = LLMTransientError(
        "private exception message: API_KEY=secret-key Authorization: Bearer secret-token"
    )
    error.http_status_code = 503
    error.provider_status = "UNAVAILABLE"
    error.error_category = "transient"
    error.provider_exception_class = "ServiceUnavailableError"
    error.provider_exception_category = "api_transient"
    error.attempt_count = 2
    service = FakeRAGService()
    service.error = error
    override_dependencies(service=service)
    caplog.set_level("WARNING", logger="app.api.chat")

    response = client.post(
        "/api/chat",
        json={"question": "private question with PDF excerpt"},
    )

    record = next(
        record
        for record in caplog.records
        if record.name == "app.api.chat"
        and record.getMessage().startswith("Chat provider request failed:")
    )
    assert record.getMessage() == (
        "Chat provider request failed: http_status_code=503 "
        "provider_status=UNAVAILABLE error_category=transient "
        "provider_exception_class=ServiceUnavailableError attempt_count=2 "
        "failure_stage=unknown provider_exception_category=api_transient"
    )
    assert {
        "http_status_code": record.http_status_code,
        "provider_status": record.provider_status,
        "error_category": record.error_category,
        "provider_exception_class": record.provider_exception_class,
        "attempt_count": record.attempt_count,
        "provider_exception_category": record.provider_exception_category,
    } == {
        "http_status_code": 503,
        "provider_status": "UNAVAILABLE",
        "error_category": "transient",
        "provider_exception_class": "ServiceUnavailableError",
        "attempt_count": 2,
        "provider_exception_category": "api_transient",
    }
    assert response.status_code == 503
    assert response.json() == {
        "detail": "The answer service is temporarily unavailable."
    }
    assert "secret-key" not in caplog.text
    assert "secret-token" not in caplog.text
    assert "private exception message" not in caplog.text
    assert "private question" not in caplog.text
    assert "PDF excerpt" not in caplog.text


def test_provider_failure_logs_safe_fallbacks_for_missing_metadata(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = FakeRAGService()
    service.error = LLMTransientError("private provider details")
    override_dependencies(service=service)
    caplog.set_level("WARNING", logger="app.api.chat")

    response = client.post("/api/chat", json={"question": "Question"})

    record = next(
        record
        for record in caplog.records
        if record.name == "app.api.chat"
        and record.getMessage().startswith("Chat provider request failed:")
    )
    assert record.getMessage() == (
        "Chat provider request failed: http_status_code=0 "
        "provider_status=unknown error_category=unknown "
        "provider_exception_class=unknown attempt_count=0 failure_stage=unknown"
        " provider_exception_category=unknown"
    )
    assert {
        "http_status_code": record.http_status_code,
        "provider_status": record.provider_status,
        "error_category": record.error_category,
        "provider_exception_class": record.provider_exception_class,
        "attempt_count": record.attempt_count,
    } == {
        "http_status_code": 0,
        "provider_status": "unknown",
        "error_category": "unknown",
        "provider_exception_class": "unknown",
        "attempt_count": 0,
    }
    assert response.status_code == 503
    assert response.json() == {
        "detail": "The answer service is temporarily unavailable."
    }
    assert "private provider details" not in caplog.text


@pytest.mark.parametrize(
    "failure_stage",
    [
        "GEMINI_RESPONSE_EXTRACTION",
        "GEMINI_JSON_DECODING",
        "GEMINI_SCHEMA_VALIDATION",
        "GEMINI_FIELD_TYPE_VALIDATION",
        "GEMINI_JSON_PARSING",
        "RAG_SOURCE_VALIDATION",
    ],
)
def test_chat_logs_allowlisted_malformed_response_stage(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    failure_stage: str,
) -> None:
    error = LLMMalformedResponseError("synthetic private response content")
    error.failure_stage = failure_stage
    service = FakeRAGService()
    service.error = error
    override_dependencies(service=service)
    caplog.set_level("WARNING", logger="app.api.chat")

    response = client.post("/api/chat", json={"question": "private question"})

    record = next(
        record
        for record in caplog.records
        if record.name == "app.api.chat"
        and record.getMessage().startswith("Chat provider request failed:")
    )
    assert f"failure_stage={failure_stage}" in record.getMessage()
    assert response.status_code == 502
    assert response.json() == {
        "detail": "The answer service returned an invalid response."
    }
    assert "synthetic private response content" not in caplog.text
    assert "private question" not in caplog.text


def test_fake_provider_remains_default_without_network_calls(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "LLM_PROVIDER", "fake")

    class EmptySearch:
        def search(self, owner_id: str, question: str, top_k: int = 5):
            return []

    factory = RAGServiceFactory(semantic_search_factory=EmptySearch)
    app.dependency_overrides[get_development_owner_id] = lambda: "trusted-owner"
    app.dependency_overrides[get_rag_service] = factory.get

    response = client.post("/api/chat", json={"question": "No stored material"})
    service = factory.get()

    assert isinstance(service.llm_provider, FakeLLMProvider)
    assert response.status_code == 200
    assert response.json()["insufficient_evidence"] is True
    assert service.llm_provider.requests == []
