from io import BytesIO
import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from app import config
from app import documents
from app.main import app
from app.repositories.interfaces import NewDocument, RepositoryOperationError
from app.repositories.sqlite_documents import SQLiteDocumentRepository
from app.storage.interfaces import StorageOperationError
from app.storage.local_filesystem import LocalFilesystemStorage
from app.services.document_service import DocumentService
from app.services import dependencies
from app.services.embeddings.service import EmbeddingService
from app.services.llm.fake import FakeLLMProvider
from app.services.llm.interfaces import LLMResponse
from app.services.rag_service import RAGService
from app.services.semantic_search import SemanticSearchService
from app.services.vector_index import VectorIndexService

client = TestClient(app)


def make_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def make_text_pdf(page_texts: list[str | None]) -> bytes:
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in page_texts:
        page = writer.add_blank_page(width=72, height=72)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font}
                )
            }
        )
        if text is not None:
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET".encode())
            page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class DeterministicEmbeddingProvider:
    dimension = 384
    model_identifier = "test-embedding-model"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, *([0.0] * (self.dimension - 1))] for _ in texts]


def make_embedding_service() -> EmbeddingService:
    return EmbeddingService(DeterministicEmbeddingProvider())


@pytest.fixture(autouse=True)
def use_temporary_document_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "DATABASE_PATH", tmp_path / "learnmate.sqlite3")
    monkeypatch.setattr(config, "DOCUMENT_STORAGE_ROOT", tmp_path / "documents")
    monkeypatch.setattr(config, "DEVELOPMENT_OWNER_ID", "test-owner")
    embedding_service = make_embedding_service()
    monkeypatch.setattr(
        dependencies,
        "get_embedding_service",
        lambda: embedding_service,
    )


def upload_pdf(
    content: bytes,
    filename: str = "notes.pdf",
    content_type: str = "application/pdf",
    idempotency_key: str | None = None,
):
    return client.post(
        "/api/documents/upload",
        headers={"Idempotency-Key": idempotency_key or str(uuid4())},
        files={"file": (filename, content, content_type)},
    )


def test_upload_stores_pdf_metadata_and_extracted_pages() -> None:
    content = make_text_pdf(["First page text"])
    response = upload_pdf(content)

    assert response.status_code == 200
    result = response.json()
    assert result == {
        "id": result["id"],
        "filename": "notes.pdf",
        "uploaded_at": result["uploaded_at"],
        "size_bytes": len(content),
        "status": "stored",
        "page_count": 1,
        "text_status": "text_found",
    }
    assert (
        config.DOCUMENT_STORAGE_ROOT / f"doc-{result['id']}.pdf"
    ).read_bytes() == content
    listing = client.get("/api/documents")
    assert listing.json()["total_count"] == 1
    assert listing.json()["documents"] == [
        {
            "id": result["id"],
            "filename": "notes.pdf",
            "uploaded_at": result["uploaded_at"],
            "size_bytes": len(content),
            "page_count": 1,
            "text_status": "text_found",
        }
    ]
    assert client.get(f"/api/documents/{result['id']}/pages").json() == {
        "document_id": result["id"],
        "pages": [{"page_number": 1, "text": "First page text"}],
    }


def test_upload_preserves_page_numbers_across_multiple_pages() -> None:
    content = make_text_pdf(["First page", "Second page"])
    response = upload_pdf(content)

    assert response.status_code == 200
    result = response.json()
    assert client.get(f"/api/documents/{result['id']}/pages").json()["pages"] == [
        {"page_number": 1, "text": "First page"},
        {"page_number": 2, "text": "Second page"},
    ]


def test_upload_persists_and_returns_cleaned_page_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Page:
        def extract_text(self) -> str:
            return "  Clean   this\ttext  "

    class Reader:
        is_encrypted = False
        pages = [Page()]

        def __init__(self, file: object, **kwargs: object) -> None:
            self.file = file
            self.options = kwargs

    monkeypatch.setattr(documents, "PdfReader", Reader)
    response = upload_pdf(make_pdf())

    assert response.status_code == 200
    result = response.json()
    assert result["text_status"] == "text_found"
    assert client.get(f"/api/documents/{result['id']}/pages").json()["pages"] == [
        {"page_number": 1, "text": "Clean this text"}
    ]


def test_upload_returns_empty_text_for_blank_page() -> None:
    content = make_pdf()
    response = upload_pdf(content)

    assert response.status_code == 200
    assert response.json() == {
        "id": response.json()["id"],
        "filename": "notes.pdf",
        "size_bytes": len(content),
        "uploaded_at": response.json()["uploaded_at"],
        "status": "stored",
        "page_count": 1,
        "text_status": "no_text_found",
    }
    assert client.get(
        f"/api/documents/{response.json()['id']}/pages"
    ).json()["pages"] == [{"page_number": 1, "text": ""}]


def test_upload_reports_pdf_with_no_selectable_text() -> None:
    content = make_text_pdf([None, None])
    response = upload_pdf(content, "scanned.pdf")

    assert response.status_code == 200
    assert response.json()["page_count"] == 2
    assert response.json()["text_status"] == "no_text_found"
    assert client.get(
        f"/api/documents/{response.json()['id']}/pages"
    ).json()["pages"] == [
        {"page_number": 1, "text": ""},
        {"page_number": 2, "text": ""},
    ]


def test_upload_reports_partially_extractable_pdf() -> None:
    content = make_text_pdf(["Readable text", None, "More text"])
    response = upload_pdf(content, "partial.pdf")

    assert response.status_code == 200
    assert response.json()["text_status"] == "partial_text"
    assert client.get(
        f"/api/documents/{response.json()['id']}/pages"
    ).json()["pages"] == [
        {"page_number": 1, "text": "Readable text"},
        {"page_number": 2, "text": ""},
        {"page_number": 3, "text": "More text"},
    ]


def test_upload_rejects_non_pdf_filename() -> None:
    response = upload_pdf(b"plain text content", "notes.txt", "text/plain")

    assert response.status_code == 415
    assert response.json() == {"detail": "Only PDF files are accepted."}


def test_upload_rejects_pdf_extension_without_pdf_content() -> None:
    response = upload_pdf(b"%PDF-1.7 not actually a PDF")

    assert response.status_code == 415
    assert response.json() == {"detail": "The uploaded file is not a valid PDF."}


def test_upload_reports_safe_error_for_unreadable_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = make_text_pdf(["Readable text"])

    class UnreadablePage:
        def extract_text(self) -> str:
            raise ValueError("sensitive parser detail")

    class UnreadableReader:
        is_encrypted = False
        pages = [UnreadablePage()]

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    monkeypatch.setattr(documents, "PdfReader", UnreadableReader)
    response = upload_pdf(content)

    assert response.status_code == 415
    assert response.json() == {
        "detail": "The PDF could not be read for text extraction."
    }


def test_upload_rejects_missing_pdf_signature() -> None:
    response = upload_pdf(b"not a PDF")

    assert response.status_code == 415
    assert response.json() == {
        "detail": "The uploaded file does not contain a PDF signature."
    }


def test_upload_rejects_empty_file() -> None:
    response = upload_pdf(b"", "empty.pdf")

    assert response.status_code == 400
    assert response.json() == {"detail": "The uploaded file is empty."}


def test_upload_rejects_pdf_larger_than_limit() -> None:
    content = b"%PDF-" + b"0" * documents.MAX_DOCUMENT_SIZE
    response = upload_pdf(content, "large.pdf")

    assert response.status_code == 413
    assert response.json() == {"detail": "PDF files must be 10 MB or smaller."}


def test_upload_accepts_pdf_at_size_limit() -> None:
    content = make_pdf()
    content += b" " * (documents.MAX_DOCUMENT_SIZE - len(content))
    response = upload_pdf(content, "limit.pdf")

    assert response.status_code == 200
    assert response.json()["size_bytes"] == documents.MAX_DOCUMENT_SIZE


def test_upload_rejects_incorrect_content_type() -> None:
    response = upload_pdf(make_pdf(), "notes.pdf", "text/plain")

    assert response.status_code == 415
    assert response.json() == {
        "detail": "The content type must be application/pdf."
    }


def test_upload_requires_file() -> None:
    response = client.post(
        "/api/documents/upload",
        headers={"Idempotency-Key": str(uuid4())},
    )

    assert response.status_code == 422


def test_same_idempotency_key_returns_existing_document() -> None:
    content = make_text_pdf(["Stable content"])
    first = upload_pdf(content, idempotency_key="retry-key")
    second = upload_pdf(content, idempotency_key="retry-key")

    assert first.status_code == second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert len(client.get("/api/documents").json()["documents"]) == 1
    assert len(list(config.DOCUMENT_STORAGE_ROOT.glob("*.pdf"))) == 1
    repository = SQLiteDocumentRepository(config.DATABASE_PATH)
    assert len(repository.get_owner_embeddings("test-owner")) == 1
    assert repository.get_owner_index_generation("test-owner") == 1


def test_upload_indexes_chunks_and_search_returns_page_citations() -> None:
    repository = SQLiteDocumentRepository(config.DATABASE_PATH)
    embedding_service = make_embedding_service()
    vector_index = VectorIndexService(repository)
    semantic_search = SemanticSearchService(embedding_service, vector_index)

    assert semantic_search.search("test-owner", "assignment question") == []
    response = upload_pdf(
        make_text_pdf(["First page material", "Second page material"]),
        "assignment.pdf",
    )

    assert response.status_code == 200
    document_id = response.json()["id"]
    stored_chunks = repository.get_owner_embeddings("test-owner")
    assert len(stored_chunks) == 1
    assert stored_chunks[0].document_id == document_id
    assert stored_chunks[0].page_numbers == (1, 2)
    assert repository.get_owner_index_generation("test-owner") == 1

    results = semantic_search.search("test-owner", "assignment question")
    assert len(results) == 1
    assert results[0].document_id == document_id
    assert results[0].document_name == "assignment.pdf"
    assert results[0].page_numbers == (1, 2)
    assert vector_index._cache["test-owner"].index.ntotal == 1

    rag = RAGService(
        semantic_search,
        FakeLLMProvider(
            LLMResponse(
                answer="The answer is in both pages.",
                source_ids=("S1",),
                insufficient_evidence=False,
            )
        ),
    )
    answer = rag.answer("test-owner", "assignment question")
    assert answer.citations[0].document_id == document_id
    assert answer.citations[0].document_name == "assignment.pdf"
    assert answer.citations[0].page_numbers == (1, 2)


def test_failed_embedding_does_not_publish_or_retain_uploaded_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_embedding(self: EmbeddingService, chunks: list[object]) -> list[object]:
        raise RuntimeError("injected embedding failure")

    monkeypatch.setattr(EmbeddingService, "embed_chunks", fail_embedding)
    response = upload_pdf(make_text_pdf(["Do not publish this document"]))

    assert response.status_code == 500
    assert response.json() == {
        "detail": "The uploaded document could not be saved."
    }
    repository = SQLiteDocumentRepository(config.DATABASE_PATH)
    assert client.get("/api/documents").json()["documents"] == []
    assert repository.get_owner_embeddings("test-owner") == []
    assert repository.get_owner_index_generation("test-owner") == 0
    assert list(config.DOCUMENT_STORAGE_ROOT.iterdir()) == []


def test_upload_search_is_isolated_by_owner() -> None:
    first = upload_pdf(make_text_pdf(["First owner's private material"]), "first.pdf")
    from app.services.dependencies import get_development_owner_id

    app.dependency_overrides[get_development_owner_id] = lambda: "other-owner"
    try:
        second = upload_pdf(
            make_text_pdf(["Second owner's private material"]), "second.pdf"
        )
    finally:
        app.dependency_overrides.pop(get_development_owner_id, None)

    assert first.status_code == second.status_code == 200
    repository = SQLiteDocumentRepository(config.DATABASE_PATH)
    semantic_search = SemanticSearchService(
        make_embedding_service(),
        VectorIndexService(repository),
    )
    first_results = semantic_search.search("test-owner", "private material")
    second_results = semantic_search.search("other-owner", "private material")

    assert {result.document_id for result in first_results} == {first.json()["id"]}
    assert {result.document_id for result in second_results} == {second.json()["id"]}


def test_reusing_idempotency_key_for_different_content_conflicts() -> None:
    first = upload_pdf(make_text_pdf(["one"]), idempotency_key="same-key")
    second = upload_pdf(make_text_pdf(["two"]), idempotency_key="same-key")

    assert first.status_code == 200
    assert second.status_code == 409
    assert len(client.get("/api/documents").json()["documents"]) == 1
    assert len(list(config.DOCUMENT_STORAGE_ROOT.glob("*.pdf"))) == 1


def test_retry_during_pending_upload_does_not_remove_original_staged_file() -> None:
    content = make_text_pdf(["in progress"])
    repository = SQLiteDocumentRepository(config.DATABASE_PATH)
    storage = LocalFilesystemStorage(config.DOCUMENT_STORAGE_ROOT)
    document_id = str(uuid4())
    staged = storage.stage(BytesIO(content), document_id)
    extraction = documents.extract_pdf_text(
        documents.validate_pdf_upload(
            "pending.pdf", "application/pdf", BytesIO(content)
        ).reader
    )
    repository.create_pending(
        NewDocument(
            id=document_id,
            owner_id="test-owner",
            filename="pending.pdf",
            storage_key=staged.storage_key,
            staging_key=staged.staging_key,
            uploaded_at="2026-10-31T10:00:00+00:00",
            size_bytes=staged.size_bytes,
            page_count=extraction["page_count"],
            text_status=extraction["text_status"],
            idempotency_key="pending-key",
            content_sha256=staged.content_sha256,
        ),
        extraction["pages"],
    )

    response = upload_pdf(content, idempotency_key="pending-key")

    assert response.status_code == 409
    assert (config.DOCUMENT_STORAGE_ROOT / staged.staging_key).exists()
    assert repository.find_by_idempotency_key(
        "test-owner", "pending-key"
    ).status == "pending"


def test_duplicate_original_filenames_do_not_overwrite_files() -> None:
    first = upload_pdf(make_text_pdf(["first"]), "duplicate.pdf")
    second = upload_pdf(make_text_pdf(["second"]), "duplicate.pdf")

    assert first.status_code == second.status_code == 200
    assert first.json()["id"] != second.json()["id"]
    assert len(list(config.DOCUMENT_STORAGE_ROOT.glob("*.pdf"))) == 2


def test_document_pages_are_not_accessible_to_another_owner() -> None:
    result = upload_pdf(make_text_pdf(["private text"])).json()
    from app.api.documents import get_development_owner_id

    app.dependency_overrides[get_development_owner_id] = lambda: "other-owner"
    try:
        assert client.get(f"/api/documents/{result['id']}/pages").status_code == 404
        listing = client.get("/api/documents").json()
        assert listing["documents"] == []
        assert listing["total_count"] == 0
    finally:
        app.dependency_overrides.pop(get_development_owner_id, None)


def test_missing_document_pages_return_not_found() -> None:
    response = client.get("/api/documents/not-a-document/pages")

    assert response.status_code == 404


def test_upload_promotion_failure_removes_staging_file_and_pending_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_promotion(self: LocalFilesystemStorage, staged: object) -> None:
        raise StorageOperationError("filesystem detail")

    monkeypatch.setattr(LocalFilesystemStorage, "promote", fail_promotion)
    response = upload_pdf(make_text_pdf(["persist me"]))

    assert response.status_code == 500
    assert response.json() == {
        "detail": "The uploaded document could not be saved."
    }
    assert list(config.DOCUMENT_STORAGE_ROOT.iterdir()) == []
    assert client.get("/api/documents").json()["documents"] == []


def test_database_finalize_failure_cleans_promoted_file_and_pending_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_finalize(
        self: SQLiteDocumentRepository,
        owner_id: str,
        document_id: str,
        model_name: str,
        chunks: list[object],
    ) -> None:
        raise RepositoryOperationError("database detail")

    monkeypatch.setattr(
        SQLiteDocumentRepository, "replace_document_embeddings", fail_finalize
    )
    response = upload_pdf(make_text_pdf(["persist me"]))

    assert response.status_code == 500
    assert list(config.DOCUMENT_STORAGE_ROOT.iterdir()) == []
    assert client.get("/api/documents").json()["documents"] == []


def test_additive_migration_preserves_existing_rows_and_files(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    documents_directory = tmp_path / "documents"
    documents_directory.mkdir()
    legacy_bytes = b"existing legacy file"
    (documents_directory / "legacy.pdf").write_bytes(legacy_bytes)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL UNIQUE,
                uploaded_at TEXT NOT NULL,
                size_bytes INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO documents
            VALUES ('legacy-id', 'legacy.pdf', 'legacy.pdf', '2025-01-01T00:00:00+00:00', ?)
            """,
            (len(legacy_bytes),),
        )

    repository = SQLiteDocumentRepository(database_path)
    repository.initialize()
    repository.initialize()

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT id, filename, stored_filename, size_bytes, owner_id, status
            FROM documents WHERE id = 'legacy-id'
            """
        ).fetchone()
        versions = [
            item[0]
            for item in connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
        ]
    assert row == (
        "legacy-id",
        "legacy.pdf",
        "legacy.pdf",
        len(legacy_bytes),
        None,
        "ready",
    )
    assert versions == [1, 2, 3, 4, 5, 6]
    assert (documents_directory / "legacy.pdf").read_bytes() == legacy_bytes
    assert repository.list_documents("test-owner", 20, None) == ([], None, 0)


def test_document_listing_uses_owner_scoped_cursor_pagination() -> None:
    for number in range(3):
        response = upload_pdf(
            make_text_pdf([f"page {number}"]),
            idempotency_key=f"pagination-{number}",
        )
        assert response.status_code == 200

    first_page = client.get("/api/documents?limit=2").json()
    assert len(first_page["documents"]) == 2
    assert first_page["next_cursor"]
    assert first_page["total_count"] == 3
    second_page = client.get(
        f"/api/documents?limit=2&cursor={first_page['next_cursor']}"
    ).json()
    assert len(second_page["documents"]) == 1
    assert second_page["next_cursor"] is None
    assert second_page["total_count"] == 3


def test_pending_upload_is_recovered_from_staged_file(tmp_path: Path) -> None:
    repository = SQLiteDocumentRepository(config.DATABASE_PATH)
    storage = LocalFilesystemStorage(config.DOCUMENT_STORAGE_ROOT)
    content = make_text_pdf(["recovered content"])
    document_id = str(uuid4())
    staged = storage.stage(BytesIO(content), document_id)
    extracted = documents.extract_pdf_text(
        documents.validate_pdf_upload(
            "recovered.pdf", "application/pdf", BytesIO(content)
        ).reader
    )
    pending = NewDocument(
        id=document_id,
        owner_id="test-owner",
        filename="recovered.pdf",
        storage_key=staged.storage_key,
        staging_key=staged.staging_key,
        uploaded_at="2026-10-31T10:00:00+00:00",
        size_bytes=staged.size_bytes,
        page_count=extracted["page_count"],
        text_status=extracted["text_status"],
        idempotency_key="recovery-key",
        content_sha256=staged.content_sha256,
    )
    repository.create_pending(pending, extracted["pages"])

    DocumentService(repository, storage, make_embedding_service()).recover_pending_uploads()

    recovered = repository.find_by_idempotency_key("test-owner", "recovery-key")
    assert recovered.status == "ready"
    assert len(repository.get_owner_embeddings("test-owner")) == 1
    assert repository.get_owner_index_generation("test-owner") == 1
    assert (config.DOCUMENT_STORAGE_ROOT / staged.storage_key).read_bytes() == content
    assert not (config.DOCUMENT_STORAGE_ROOT / staged.staging_key).exists()


def test_page_insert_failure_rolls_back_document_record(tmp_path: Path) -> None:
    repository = SQLiteDocumentRepository(tmp_path / "documents.sqlite3")
    document = NewDocument(
        id=str(uuid4()),
        owner_id="owner",
        filename="partial.pdf",
        storage_key=f"doc-{uuid4()}.pdf",
        staging_key=f".staging-{uuid4()}.part",
        uploaded_at="2026-01-01T00:00:00+00:00",
        size_bytes=123,
        page_count=2,
        text_status="text_found",
        idempotency_key="partial-insert",
        content_sha256="hash",
    )
    with pytest.raises(RepositoryOperationError):
        repository.create_pending(
            document,
            [
                {"page_number": 1, "text": "one"},
                {"page_number": 1, "text": "duplicate"},
            ],
        )
    assert repository.find_by_idempotency_key("owner", "partial-insert") is None
