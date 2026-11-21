import json
from concurrent.futures import Executor, Future
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.documents import get_development_owner_id
from app.main import app
from app.repositories.interfaces import NewDocument
from app.repositories.sqlite_documents import SQLiteDocumentRepository
from app.repositories.sqlite_summaries import SQLiteSummaryRepository
from app.services.dependencies import get_document_summary_job_service
from app.services.document_summary import DocumentSummaryService
from app.services.document_summary_jobs import DocumentSummaryJobService
from app.services.llm.interfaces import LLMRequest, LLMResponse


class DeferredExecutor(Executor):
    def __init__(self) -> None:
        self.pending: list[tuple[Future[object], object, tuple[object, ...]]] = []

    def submit(self, fn, /, *args, **kwargs):
        future: Future[object] = Future()
        self.pending.append((future, fn, args))
        return future

    def run_next(self) -> None:
        future, fn, args = self.pending.pop(0)
        if future.set_running_or_notify_cancel():
            future.set_result(fn(*args))


class APIProvider:
    def generate(self, request: LLMRequest) -> LLMResponse:
        sources = json.loads(
            request.user_prompt[request.user_prompt.index('[{"source_id"') :]
        )
        source_ids = tuple(source["source_id"] for source in sources)
        if request.question == "Write the final exam-oriented document summary.":
            answer = (
                "## Topic-wise explanations\n"
                "### Topic\nSupported explanation.\n\n"
                "## Final revision\n- Supported recall point."
            )
        else:
            answer = "### Notes\nSupported detail."
        return LLMResponse(answer, source_ids, False)


def create_document(
    repository: SQLiteDocumentRepository, document_id: str, owner_id: str
) -> None:
    document = NewDocument(
        id=document_id,
        owner_id=owner_id,
        filename=f"{document_id}.pdf",
        storage_key=f"doc-{document_id}.pdf",
        staging_key=f".staging-{document_id}.part",
        uploaded_at="2026-10-31T00:00:00+00:00",
        size_bytes=1,
        page_count=1,
        text_status="text_found",
        idempotency_key=f"{document_id}-{owner_id}",
        content_sha256=f"content-{document_id}",
    )
    repository.create_pending(
        document, [{"page_number": 1, "text": "A source-backed definition."}]
    )
    repository.mark_ready(owner_id, document_id)


def make_job_service(
    tmp_path: Path, executor: Executor
) -> DocumentSummaryJobService:
    database_path = tmp_path / "summary-api.sqlite3"
    documents = SQLiteDocumentRepository(database_path)
    create_document(documents, "doc-1", "owner-1")
    create_document(documents, "doc-2", "owner-2")
    return DocumentSummaryJobService(
        SQLiteSummaryRepository(database_path),
        documents,
        DocumentSummaryService(APIProvider()),
        executor=executor,
    )


def test_summary_endpoint_starts_job_polls_progress_and_returns_summary(
    tmp_path: Path,
) -> None:
    executor = DeferredExecutor()
    service = make_job_service(tmp_path, executor)
    app.dependency_overrides[get_development_owner_id] = lambda: "owner-1"
    app.dependency_overrides[get_document_summary_job_service] = lambda: service
    client = TestClient(app)
    try:
        started = client.post("/api/documents/doc-1/summary")
        assert started.status_code == 202
        start_body = started.json()
        assert start_body["status"] == "queued"
        assert start_body["phase"] == "queued"
        assert start_body["result"] is None

        pending = client.get(f"/api/documents/doc-1/summary/{start_body['id']}")
        assert pending.status_code == 200
        assert pending.json()["status"] == "queued"

        executor.run_next()
        completed = client.get(f"/api/documents/doc-1/summary/{start_body['id']}")
        assert completed.status_code == 200
        body = completed.json()
        assert body["status"] == "completed"
        assert body["phase"] == "completed"
        assert body["progress_current"] == body["progress_total"]
        assert body["result"]["page_references"] == [1]
        assert "## Final revision" in body["result"]["summary_markdown"]
    finally:
        app.dependency_overrides.pop(get_development_owner_id, None)
        app.dependency_overrides.pop(get_document_summary_job_service, None)


def test_summary_endpoints_enforce_document_and_job_owner_isolation(
    tmp_path: Path,
) -> None:
    service = make_job_service(tmp_path, DeferredExecutor())
    owner = ["owner-1"]
    app.dependency_overrides[get_development_owner_id] = lambda: owner[0]
    app.dependency_overrides[get_document_summary_job_service] = lambda: service
    client = TestClient(app)
    try:
        assert client.post("/api/documents/doc-2/summary").status_code == 404
        started = client.post("/api/documents/doc-1/summary")
        assert started.status_code == 202
        job_id = started.json()["id"]

        owner[0] = "owner-2"
        assert (
            client.get(f"/api/documents/doc-1/summary/{job_id}").status_code
            == 404
        )
        assert client.post("/api/documents/doc-1/summary").status_code == 404
    finally:
        app.dependency_overrides.pop(get_development_owner_id, None)
        app.dependency_overrides.pop(get_document_summary_job_service, None)
