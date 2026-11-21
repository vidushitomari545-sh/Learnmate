from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.repositories.interfaces import NewDocument
from app.repositories.sqlite_documents import SQLiteDocumentRepository
from app.services.document_service import DocumentService
from app.services.text_chunker import chunk_text_pages
from app.storage.local_filesystem import LocalFilesystemStorage


def test_short_text_produces_one_chunk() -> None:
    chunks = chunk_text_pages(
        "document-1", [{"page_number": 1, "text": "Short text."}]
    )

    assert len(chunks) == 1
    assert chunks[0]["text"] == "Short text."
    assert chunks[0]["chunk_index"] == 0


def test_long_text_produces_multiple_chunks() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 4
    chunks = chunk_text_pages(
        "document-1", [{"page_number": 1, "text": text}], chunk_size=30, overlap=5
    )

    assert len(chunks) > 1
    assert all(0 < len(chunk["text"]) <= 30 for chunk in chunks)


def test_consecutive_chunks_have_configured_character_overlap() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 3
    chunks = chunk_text_pages(
        "document-1", [{"page_number": 1, "text": text}], chunk_size=20, overlap=6
    )

    assert len(chunks) > 1
    for previous, following in zip(chunks, chunks[1:]):
        assert previous["text"][-6:] == following["text"][:6]


def test_chunk_boundary_prefers_paragraph_break() -> None:
    text = "First paragraph is here.\n\nSecond paragraph follows."
    chunks = chunk_text_pages(
        "document-1", [{"page_number": 1, "text": text}], chunk_size=32, overlap=4
    )

    assert len(chunks) == 2
    assert chunks[0]["text"].endswith("\n\n")


def test_empty_input_produces_no_chunks() -> None:
    assert chunk_text_pages("document-1", []) == []


def test_empty_pdf_pages_produce_no_chunks() -> None:
    chunks = chunk_text_pages(
        "document-1",
        [
            {"page_number": 1, "text": ""},
            {"page_number": 2, "text": " \t\n "},
        ],
    )

    assert chunks == []


def test_chunk_preserves_original_page_numbers_with_empty_pages() -> None:
    chunks = chunk_text_pages(
        "document-1",
        [
            {"page_number": 1, "text": "First page."},
            {"page_number": 2, "text": ""},
            {"page_number": 3, "text": "Third page."},
        ],
    )

    assert len(chunks) == 1
    assert chunks[0]["page_numbers"] == [1, 3]
    assert chunks[0]["start_page"] == 1
    assert chunks[0]["end_page"] == 3


def test_chunk_spanning_pages_lists_each_source_page() -> None:
    chunks = chunk_text_pages(
        "document-1",
        [
            {"page_number": 4, "text": "First source page."},
            {"page_number": 5, "text": "Second source page."},
        ],
        chunk_size=100,
        overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0]["page_numbers"] == [4, 5]
    assert chunks[0]["start_page"] == 4
    assert chunks[0]["end_page"] == 5


def test_chunk_size_and_overlap_are_configurable() -> None:
    chunks = chunk_text_pages(
        "document-1",
        [{"page_number": 1, "text": "abcdefghijklmnopqrstuvwxyz"}],
        chunk_size=12,
        overlap=3,
    )

    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 12 for chunk in chunks)
    assert chunks[0]["text"][-3:] == chunks[1]["text"][:3]


def test_chunking_preserves_content_and_makes_progress() -> None:
    text = "abcdefghij " * 30
    chunks = chunk_text_pages(
        "document-1", [{"page_number": 1, "text": text}], chunk_size=24, overlap=7
    )

    assert len(chunks) > 1
    reconstructed = chunks[0]["text"] + "".join(
        chunk["text"][7:] for chunk in chunks[1:]
    )
    assert reconstructed == text.strip()


def test_chunk_records_include_unique_ids_and_document_id() -> None:
    chunks = chunk_text_pages(
        "document-xyz",
        [{"page_number": 1, "text": "abcdefghijklmnopqrstuvwxyz"}],
        chunk_size=10,
        overlap=2,
    )

    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)
    assert all(chunk["document_id"] == "document-xyz" for chunk in chunks)
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_document_service_chunks_stored_page_records(tmp_path: Path) -> None:
    repository = SQLiteDocumentRepository(tmp_path / "documents.sqlite3")
    repository.initialize()
    document = NewDocument(
        id=str(uuid4()),
        owner_id="owner-1",
        filename="notes.pdf",
        storage_key="doc-placeholder.pdf",
        staging_key=".staging-placeholder.part",
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        size_bytes=1,
        page_count=2,
        text_status="text_found",
        idempotency_key="chunk-service-test",
        content_sha256="hash",
    )
    repository.create_pending(
        document,
        [
            {"page_number": 1, "text": "First page."},
            {"page_number": 2, "text": "Second page."},
        ],
    )
    repository.mark_ready("owner-1", document.id)
    service = DocumentService(repository, LocalFilesystemStorage(tmp_path))

    chunks = service.chunk_document(
        "owner-1", document.id, chunk_size=100, overlap=10
    )

    assert chunks is not None
    assert len(chunks) == 1
    assert chunks[0]["document_id"] == document.id
    assert chunks[0]["page_numbers"] == [1, 2]
