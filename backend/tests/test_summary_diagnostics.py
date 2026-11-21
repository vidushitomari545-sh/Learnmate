import json
import logging
from collections.abc import Callable
from concurrent.futures import Executor, Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.repositories.interfaces import NewDocument
from app.repositories.sqlite_documents import SQLiteDocumentRepository
from app.repositories.sqlite_summaries import SQLiteSummaryRepository
from app.services.document_summary import (
    DocumentSummaryService,
    MalformedSummaryOutputError,
    SUMMARY_RESPONSE_CONTRACT,
    SummaryResponseDiagnostic,
)
from app.services.document_summary_jobs import DocumentSummaryJobService
from app.services.llm.interfaces import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMMalformedResponseError,
    inspect_source_ids_shape,
)
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.openai_provider import OpenAIProvider
from app.services.llm.ollama_provider import OllamaProvider


def _ollama_body(content: str) -> bytes:
    return json.dumps({"message": {"content": content}}).encode("utf-8")


def _structured_content(
    *,
    answer: object = "A grounded summary.",
    source_ids: object = ("C000001",),
    insufficient_evidence: object = False,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "source_ids": source_ids,
            "insufficient_evidence": insufficient_evidence,
        }
    )


class _OllamaParserProvider:
    def __init__(self, response_body: bytes) -> None:
        self.response_body = response_body

    def generate(self, request: LLMRequest) -> LLMResponse:
        return OllamaProvider._parse_response(
            self.response_body,
            {source.source_id for source in request.sources},
        )


class _ProviderFailure:
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise LLMProviderError("PRIVATE_MODEL_RESPONSE_AND_API_KEY_SENTINEL")


@pytest.mark.parametrize(
    ("category", "response_body", "expected_type", "source_ids_rejected"),
    [
        (
            "invalid_json",
            _ollama_body("{not valid JSON"),
            "str",
            False,
        ),
        (
            "missing_fields",
            _ollama_body('{"answer":"Summary","insufficient_evidence":false}'),
            "dict",
            False,
        ),
        (
            "field_type",
            _ollama_body(_structured_content(answer=17)),
            "int",
            False,
        ),
        (
            "source_ids",
            _ollama_body(_structured_content(source_ids=[17])),
            "list",
            True,
        ),
        (
            "source_ids",
            _ollama_body(_structured_content(source_ids=["NOT-A-SOURCE"])),
            "tuple",
            True,
        ),
        (
            "empty_answer",
            _ollama_body(_structured_content(answer="")),
            "str",
            False,
        ),
    ],
)
def test_provider_validation_diagnostic_is_allowlisted_and_has_mapping_stage(
    category: str,
    response_body: bytes,
    expected_type: str,
    source_ids_rejected: bool,
) -> None:
    service = DocumentSummaryService(_OllamaParserProvider(response_body))

    with pytest.raises(
        (LLMMalformedResponseError, MalformedSummaryOutputError)
    ) as raised:
        service.summarize_pages(
            "diagnostic-document",
            [{"page_number": 1, "text": "A source-backed statement."}],
            page_count=1,
            text_status="text_found",
        )

    diagnostic = (
        raised.value.diagnostic
        if isinstance(raised.value, MalformedSummaryOutputError)
        else raised.value.summary_diagnostic
    )
    assert diagnostic.stage == "mapping"
    assert diagnostic.category == category
    assert diagnostic.expected_contract == SUMMARY_RESPONSE_CONTRACT
    assert diagnostic.actual_response_type == expected_type
    assert diagnostic.source_ids_rejected is source_ids_rejected
    if category == "source_ids":
        shape = diagnostic.source_ids_shape
        assert shape is not None
        assert shape.container_type == expected_type
        assert shape.item_count == 1
        assert shape.non_string_member_count == (
            1 if expected_type == "list" else 0
        )
        assert shape.unknown_id_count == (1 if expected_type == "tuple" else 0)
        assert shape.normalization_ran is False


@pytest.mark.parametrize(
    (
        "source_ids",
        "expected_ids",
        "expected_count",
        "expected_non_string_count",
        "expected_empty_string_count",
        "expected_duplicate_count",
        "expected_unknown_count",
        "expected_empty",
        "normalization_ran",
    ),
    [
        (["SYN-A"], {"SYN-A"}, 1, 0, 0, 0, 0, False, True),
        (["SYN-A", 7], {"SYN-A"}, 2, 1, 0, 0, 0, False, False),
        (["SYN-A", " \t"], {"SYN-A"}, 2, 0, 1, 0, 0, False, False),
        (["SYN-A", "SYN-A"], {"SYN-A"}, 2, 0, 0, 1, 0, False, False),
        (["SYN-UNKNOWN"], {"SYN-A"}, 1, 0, 0, 0, 1, False, True),
        ([], {"SYN-A"}, 0, 0, 0, 0, 0, True, False),
    ],
    ids=[
        "valid-string",
        "non-string-member",
        "whitespace-member",
        "duplicate-member",
        "unknown-member",
        "empty-list",
    ],
)
def test_source_id_shape_diagnostics_record_counts_only(
    source_ids: list[object],
    expected_ids: set[str],
    expected_count: int,
    expected_non_string_count: int,
    expected_empty_string_count: int,
    expected_duplicate_count: int,
    expected_unknown_count: int,
    expected_empty: bool,
    normalization_ran: bool,
) -> None:
    diagnostic = inspect_source_ids_shape(
        source_ids,
        expected_ids,
        normalization_ran=normalization_ran,
    )

    assert diagnostic.container_type == "list"
    assert diagnostic.item_count == expected_count
    assert diagnostic.non_string_member_count == expected_non_string_count
    assert diagnostic.empty_string_member_count == expected_empty_string_count
    assert diagnostic.duplicate_string_member_count == expected_duplicate_count
    assert diagnostic.unknown_id_count == expected_unknown_count
    assert diagnostic.is_empty is expected_empty
    assert diagnostic.normalization_ran is normalization_ran


def test_provider_error_diagnostic_does_not_include_exception_content() -> None:
    service = DocumentSummaryService(_ProviderFailure())

    with pytest.raises(LLMProviderError) as raised:
        service.summarize_pages(
            "diagnostic-document",
            [{"page_number": 1, "text": "A source-backed statement."}],
            page_count=1,
            text_status="text_found",
        )

    diagnostic = raised.value.summary_diagnostic
    assert diagnostic.stage == "mapping"
    assert diagnostic.category == "provider_error"
    assert diagnostic.expected_contract == SUMMARY_RESPONSE_CONTRACT
    assert diagnostic.actual_response_type == "unavailable"
    assert diagnostic.source_ids_rejected is False


@pytest.mark.parametrize(
    ("parser", "response", "category", "actual_type", "source_ids_rejected"),
    [
        (
            GeminiProvider._parse_response,
            SimpleNamespace(text="{not valid JSON"),
            "invalid_json",
            "str",
            False,
        ),
        (
            OpenAIProvider._parse_response,
            SimpleNamespace(
                output_text=_structured_content(source_ids=[17])
            ),
            "source_ids",
            "list",
            True,
        ),
    ],
)
def test_non_live_provider_parsers_expose_safe_diagnostic_metadata(
    parser: Callable[[object], LLMResponse],
    response: object,
    category: str,
    actual_type: str,
    source_ids_rejected: bool,
) -> None:
    with pytest.raises(LLMMalformedResponseError) as raised:
        parser(response)

    assert raised.value.diagnostic_category == category
    assert raised.value.actual_response_type == actual_type
    assert raised.value.source_ids_rejected is source_ids_rejected


class _StageFailureProvider:
    def __init__(self, fail_on: str) -> None:
        self.fail_on = fail_on

    def generate(self, request: LLMRequest) -> LLMResponse:
        if request.question == self.fail_on:
            raise LLMMalformedResponseError(
                "PRIVATE_RESPONSE_SENTINEL",
                diagnostic_category="invalid_json",
                actual_response_type="str",
            )
        source_ids = tuple(source.source_id for source in request.sources)
        if request.question == "Summarize the supplied PDF section.":
            answer = "### Notes\n" + "Supported detail. " * 25
            return LLMResponse(answer, source_ids, False)
        return LLMResponse(
            "## Topic-wise explanations\n"
            "### Supported topic\nExplanation from the sources.\n\n"
            "## Final revision\n- Recall the supported idea.",
            source_ids,
            False,
        )


@pytest.mark.parametrize(
    ("fail_on", "expected_stage", "page_count", "text_size"),
    [
        ("Combine the supplied section summaries.", "reducing", 12, 1800),
        ("Write the final exam-oriented document summary.", "finalizing", 1, 100),
    ],
)
def test_provider_diagnostic_records_reduce_and_final_stages(
    fail_on: str,
    expected_stage: str,
    page_count: int,
    text_size: int,
) -> None:
    service = DocumentSummaryService(
        _StageFailureProvider(fail_on),
        max_context_tokens=1200,
        max_context_bytes=5000,
        max_output_tokens=200,
    )
    pages = [
        {"page_number": number, "text": f"Page {number} " + "x" * text_size}
        for number in range(1, page_count + 1)
    ]

    with pytest.raises(LLMMalformedResponseError) as raised:
        service.summarize_pages(
            "diagnostic-document",
            pages,
            page_count=page_count,
            text_status="text_found",
        )

    assert raised.value.summary_diagnostic.stage == expected_stage


class _UnusedExecutor(Executor):
    def submit(self, fn, /, *args, **kwargs):
        raise AssertionError("This test runs the job directly.")


class _StoredPages:
    def __init__(self, repository: SQLiteDocumentRepository) -> None:
        self.repository = repository

    def get_document_pages(self, owner_id: str, document_id: str):
        return self.repository.get_document_pages(owner_id, document_id)


class _SensitiveFailureProvider:
    def generate(self, request: LLMRequest) -> LLMResponse:
        raise LLMMalformedResponseError(
            "PDF_PRIVATE_TEXT RESPONSE_PRIVATE_TEXT API_KEY_PRIVATE_SENTINEL "
            "PRIVATE_RESPONSE_SENTINEL SOURCE_ID_PRIVATE_SENTINEL",
            diagnostic_category="source_ids",
            actual_response_type="list",
            source_ids_rejected=True,
            source_ids_shape=inspect_source_ids_shape(
                ["SOURCE_ID_PRIVATE_SENTINEL", 7],
                {source.source_id for source in request.sources},
                normalization_ran=False,
            ),
        )


def _create_document(repository: SQLiteDocumentRepository) -> None:
    document = NewDocument(
        id="diagnostic-doc",
        owner_id="diagnostic-owner",
        filename="diagnostic.pdf",
        storage_key="diagnostic.pdf",
        staging_key=".diagnostic.part",
        uploaded_at="2026-11-20T00:00:00+00:00",
        size_bytes=1,
        page_count=1,
        text_status="text_found",
        idempotency_key="diagnostic-key",
        content_sha256="diagnostic-content",
    )
    repository.create_pending(
        document,
        [{"page_number": 1, "text": "PDF_PRIVATE_TEXT_SENTINEL"}],
    )
    repository.mark_ready("diagnostic-owner", "diagnostic-doc")


def test_job_logs_only_allowlisted_diagnostic_fields(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    database_path = tmp_path / "summary-diagnostics.sqlite3"
    document_repository = SQLiteDocumentRepository(database_path)
    summary_repository = SQLiteSummaryRepository(database_path)
    _create_document(document_repository)
    created = summary_repository.create_summary_job(
        "diagnostic-owner", "diagnostic-doc", 1
    )
    assert created is not None
    _, job, _ = created
    service = DocumentSummaryJobService(
        summary_repository,
        _StoredPages(document_repository),
        DocumentSummaryService(_SensitiveFailureProvider()),
        executor=_UnusedExecutor(),
    )

    with caplog.at_level(logging.WARNING):
        service._run_job(job.id, job.owner_id, job.document_id)

    record = summary_repository.get_summary_job(
        job.owner_id, job.document_id, job.id
    )
    assert record is not None
    assert record.status == "failed"
    assert record.error_code == "malformed_output"
    assert f"job_id={job.id}" in caplog.text
    assert "stage=mapping" in caplog.text
    assert "category=source_ids" in caplog.text
    assert f"expected_contract={SUMMARY_RESPONSE_CONTRACT}" in caplog.text
    assert "actual_response_type=list" in caplog.text
    assert "source_ids_rejected=True" in caplog.text
    assert "container_type=list" in caplog.text
    assert "item_count=2" in caplog.text
    assert "non_string_member_count=1" in caplog.text
    assert "empty_or_whitespace_string_member_count=0" in caplog.text
    assert "duplicate_string_member_count=0" in caplog.text
    assert "unknown_id_count=1" in caplog.text
    assert "list_empty=False" in caplog.text
    assert "normalization_ran=False" in caplog.text
    assert "PDF_PRIVATE_TEXT" not in caplog.text
    assert "RESPONSE_PRIVATE_TEXT" not in caplog.text
    assert "API_KEY_PRIVATE_SENTINEL" not in caplog.text
    assert "PRIVATE_RESPONSE_SENTINEL" not in caplog.text
    assert "SOURCE_ID_PRIVATE_SENTINEL" not in caplog.text


def test_summary_diagnostic_rejects_non_allowlisted_values() -> None:
    with pytest.raises(ValueError, match="Invalid summary response diagnostic"):
        SummaryResponseDiagnostic(
            stage="unknown",
            category="unexpected",
            expected_contract=SUMMARY_RESPONSE_CONTRACT,
            actual_response_type="str",
            source_ids_rejected=False,
        )
