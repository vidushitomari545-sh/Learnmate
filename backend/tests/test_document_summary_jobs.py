import json
import sqlite3
from concurrent.futures import Executor, Future
from pathlib import Path

import pytest

from app.repositories.interfaces import NewDocument
from app.repositories.sqlite_documents import SQLiteDocumentRepository
from app.repositories.sqlite_summaries import SQLiteSummaryRepository
from app.services.document_summary import (
    DocumentSummaryService,
)
from app.services.document_summary_jobs import DocumentSummaryJobService
from app.services.llm.interfaces import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)


class ImmediateExecutor(Executor):
    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as error:
                future.set_exception(error)
        return future


class SummaryProvider:
    def __init__(self) -> None:
        self.calls = 0
        self.error: LLMProviderError | None = None
        self.malformed_final = False

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.error is not None:
            raise self.error
        source_json = request.user_prompt[
            request.user_prompt.index('[{"source_id"') :
        ]
        sources = json.loads(source_json)
        source_ids = tuple(source["source_id"] for source in sources)
        if request.question == "Write the final exam-oriented document summary.":
            answer = (
                "A malformed response."
                if self.malformed_final
                else "## Topic-wise explanations\n"
                "### Topic\nExplanation from the PDF.\n\n"
                "## Final revision\n- Recall the key definition."
            )
        else:
            answer = "### Notes\nSource-grounded details."
        return LLMResponse(answer, source_ids, False)


class StoredPagesService:
    def __init__(self, repository: SQLiteDocumentRepository) -> None:
        self.repository = repository

    def get_document_pages(self, owner_id: str, document_id: str):
        return self.repository.get_document_pages(owner_id, document_id)


def create_document(
    repository: SQLiteDocumentRepository,
    *,
    document_id: str = "doc-1",
    owner_id: str = "owner-1",
    pages: list[dict[str, str | int]] | None = None,
    content_sha256: str = "content-v1",
) -> None:
    stored_pages = pages or [{"page_number": 1, "text": "A useful definition."}]
    text_status = (
        "no_text_found"
        if not any(str(page["text"]).strip() for page in stored_pages)
        else (
            "partial_text"
            if any(not str(page["text"]).strip() for page in stored_pages)
            else "text_found"
        )
    )
    document = NewDocument(
        id=document_id,
        owner_id=owner_id,
        filename=f"{document_id}.pdf",
        storage_key=f"doc-{document_id}.pdf",
        staging_key=f".staging-{document_id}.part",
        uploaded_at="2026-10-31T00:00:00+00:00",
        size_bytes=1,
        page_count=len(stored_pages),
        text_status=text_status,
        idempotency_key=f"key-{document_id}-{owner_id}",
        content_sha256=content_sha256,
    )
    repository.create_pending(document, stored_pages)
    repository.mark_ready(owner_id, document_id)


def make_service(
    tmp_path: Path,
    *,
    provider: SummaryProvider | None = None,
) -> tuple[DocumentSummaryJobService, SQLiteSummaryRepository, SummaryProvider]:
    database_path = tmp_path / "summaries.sqlite3"
    documents = SQLiteDocumentRepository(database_path)
    summaries = SQLiteSummaryRepository(database_path)
    provider = provider or SummaryProvider()
    service = DocumentSummaryJobService(
        summaries,
        StoredPagesService(documents),
        DocumentSummaryService(provider),
        executor=ImmediateExecutor(),
    )
    return service, summaries, provider


def test_summary_cache_reuses_result_and_invalidates_changed_content(
    tmp_path: Path,
) -> None:
    service, repository, provider = make_service(tmp_path)
    create_document(
        SQLiteDocumentRepository(tmp_path / "summaries.sqlite3"),
    )

    first = service.start_summary("owner-1", "doc-1")
    assert first is not None
    assert first.status == "completed"
    assert first.cache_hit is False
    assert first.result is not None
    calls_after_first_summary = provider.calls

    cached = service.start_summary("owner-1", "doc-1")
    assert cached is not None
    assert cached.status == "completed"
    assert cached.cache_hit is True
    assert provider.calls == calls_after_first_summary

    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "UPDATE documents SET content_sha256 = ? WHERE id = ?",
            ("content-v2", "doc-1"),
        )
    changed = service.start_summary("owner-1", "doc-1")
    assert changed is not None
    assert changed.status == "completed"
    assert changed.cache_hit is False
    assert provider.calls > calls_after_first_summary


def test_partial_extraction_is_reported_with_all_available_page_references(
    tmp_path: Path,
) -> None:
    service, _, _ = make_service(tmp_path)
    create_document(
        SQLiteDocumentRepository(tmp_path / "summaries.sqlite3"),
        pages=[
            {"page_number": 1, "text": "A supported definition."},
            {"page_number": 2, "text": ""},
        ],
    )

    job = service.start_summary("owner-1", "doc-1")

    assert job is not None
    assert job.status == "completed"
    assert job.result is not None
    assert job.result["page_references"] == [1]
    assert job.result["pages_without_text"] == [2]
    assert job.result["text_status"] == "partial_text"


def test_no_extractable_text_fails_clearly_without_provider_calls(
    tmp_path: Path,
) -> None:
    service, _, provider = make_service(tmp_path)
    create_document(
        SQLiteDocumentRepository(tmp_path / "summaries.sqlite3"),
        pages=[{"page_number": 1, "text": "  "}],
    )

    job = service.start_summary("owner-1", "doc-1")

    assert job is not None
    assert job.status == "failed"
    assert job.error_code == "no_extractable_text"
    assert job.retryable is False
    assert provider.calls == 0


def test_provider_failure_is_retryable_and_a_new_request_can_succeed(
    tmp_path: Path,
) -> None:
    provider = SummaryProvider()
    provider.error = LLMTimeoutError("timeout")
    service, _, _ = make_service(tmp_path, provider=provider)
    create_document(SQLiteDocumentRepository(tmp_path / "summaries.sqlite3"))

    failed = service.start_summary("owner-1", "doc-1")
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_code == "provider_timeout"
    assert failed.retryable is True

    provider.error = None
    retried = service.start_summary("owner-1", "doc-1")
    assert retried is not None
    assert retried.status == "completed"
    assert retried.cache_hit is False


def test_malformed_final_summary_is_not_cached_or_returned(tmp_path: Path) -> None:
    provider = SummaryProvider()
    provider.malformed_final = True
    service, _, _ = make_service(tmp_path, provider=provider)
    create_document(SQLiteDocumentRepository(tmp_path / "summaries.sqlite3"))

    failed = service.start_summary("owner-1", "doc-1")

    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_code == "malformed_output"
    assert failed.retryable is True


def test_owner_isolation_prevents_reading_or_summarizing_another_users_document(
    tmp_path: Path,
) -> None:
    service, repository, _ = make_service(tmp_path)
    documents = SQLiteDocumentRepository(repository.database_path)
    create_document(documents, owner_id="owner-1")
    create_document(documents, document_id="doc-2", owner_id="owner-2")

    assert service.start_summary("owner-1", "doc-2") is None
    owner_one_job = service.start_summary("owner-1", "doc-1")
    assert owner_one_job is not None
    assert (
        service.get_summary_job("owner-2", "doc-1", owner_one_job.id) is None
    )


def test_startup_recovery_marks_interrupted_jobs_retryable(tmp_path: Path) -> None:
    service, repository, _ = make_service(tmp_path)
    documents = SQLiteDocumentRepository(repository.database_path)
    create_document(documents)
    queued = repository.create_summary_job("owner-1", "doc-1", 1)
    assert queued is not None
    _, job, _ = queued
    assert repository.claim_summary_job(job.id)

    pending = repository.recover_summary_jobs()

    assert pending == []
    recovered = repository.get_summary_job("owner-1", "doc-1", job.id)
    assert recovered is not None
    assert recovered.status == "failed"
    assert recovered.error_code == "interrupted"
    assert recovered.retryable is True


def test_job_progress_is_bounded_and_cached_results_include_validated_sources(
    tmp_path: Path,
) -> None:
    service, _, _ = make_service(tmp_path)
    create_document(SQLiteDocumentRepository(tmp_path / "summaries.sqlite3"))

    job = service.start_summary("owner-1", "doc-1")

    assert job is not None
    assert job.status == "completed"
    assert job.progress_current == job.progress_total
    assert job.progress_total > 0
    assert job.result is not None
    assert job.result["page_references"] == [1]
    with pytest.raises(ValueError, match="bounded"):
        service.repository.update_summary_progress(
            job.id, "mapping", current=2, total=1
        )
