import json
import struct
from pathlib import Path

import pytest

from app.repositories.interfaces import (
    DocumentNotReadyError,
    NewDocument,
    RepositoryOperationError,
)
from app.repositories.sqlite_documents import (
    EMBEDDING_DIMENSION,
    EMBEDDING_ENCODING,
    SQLiteDocumentRepository,
)
from app.services.embeddings.types import EmbeddedChunk


def make_repository(tmp_path: Path) -> SQLiteDocumentRepository:
    return SQLiteDocumentRepository(tmp_path / "learnmate.sqlite3")


def create_ready_document(
    repository: SQLiteDocumentRepository,
    document_id: str,
    owner_id: str,
) -> None:
    repository.create_pending(
        NewDocument(
            id=document_id,
            owner_id=owner_id,
            filename=f"{document_id}.pdf",
            storage_key=f"doc-{document_id}.pdf",
            staging_key=f".staging-{document_id}.part",
            uploaded_at="2026-10-31T00:00:00+00:00",
            size_bytes=1,
            page_count=3,
            text_status="extracted",
            idempotency_key=f"key-{document_id}",
            content_sha256="hash",
        ),
        [
            {"page_number": 1, "text": "first page"},
            {"page_number": 2, "text": "second page"},
            {"page_number": 3, "text": "third page"},
        ],
    )
    repository.mark_ready(owner_id, document_id)


def make_embedded_chunk(
    document_id: str,
    chunk_index: int = 0,
    *,
    chunk_id: str | None = None,
    vector: list[float] | None = None,
) -> EmbeddedChunk:
    return {
        "chunk_id": chunk_id or f"{document_id}-chunk-{chunk_index}",
        "document_id": document_id,
        "chunk_index": chunk_index,
        "text": f"Text for chunk {chunk_index}",
        "page_numbers": [2, 3],
        "start_page": 2,
        "end_page": 3,
        "embedding": vector or [1.0, *([0.0] * (EMBEDDING_DIMENSION - 1))],
    }


def test_stores_and_decodes_float32_little_endian_blob(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    create_ready_document(repository, "doc-one", "owner-one")
    vector = [0.0, 1.0, *([0.0] * (EMBEDDING_DIMENSION - 2))]
    chunk = make_embedded_chunk("doc-one", vector=vector)

    repository.replace_document_embeddings(
        "owner-one", "doc-one", "model-v1", [chunk]
    )
    stored = repository.get_owner_embeddings("owner-one")

    assert len(stored) == 1
    assert stored[0].embedding == tuple(vector)
    assert stored[0].dimension == EMBEDDING_DIMENSION
    assert stored[0].encoding == EMBEDDING_ENCODING
    assert stored[0].model_name == "model-v1"
    with repository._connect() as connection:
        row = connection.execute(
            "SELECT embedding, page_numbers_json FROM embedded_chunks"
        ).fetchone()
    assert len(row["embedding"]) == 1536
    assert row["embedding"] == struct.pack(f"<{EMBEDDING_DIMENSION}f", *vector)
    assert json.loads(row["page_numbers_json"]) == [2, 3]


@pytest.mark.parametrize(
    ("vector", "error"),
    [
        ([1.0, 0.0], "exactly 384"),
        ([float("nan"), *([0.0] * 383)], "finite numbers"),
        ([2.0, *([0.0] * 383)], "L2-normalized"),
    ],
)
def test_rejects_invalid_embedding_vectors(
    tmp_path: Path, vector: list[float], error: str
) -> None:
    repository = make_repository(tmp_path)
    create_ready_document(repository, "doc-one", "owner-one")

    with pytest.raises(ValueError, match=error):
        repository.replace_document_embeddings(
            "owner-one",
            "doc-one",
            "model-v1",
            [make_embedded_chunk("doc-one", vector=vector)],
        )


def test_rejects_chunks_for_another_document(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    create_ready_document(repository, "doc-one", "owner-one")

    with pytest.raises(ValueError, match="another document"):
        repository.replace_document_embeddings(
            "owner-one",
            "doc-one",
            "model-v1",
            [make_embedded_chunk("doc-two")],
        )


def test_persisting_pending_embeddings_atomically_makes_document_ready(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    repository.create_pending(
        NewDocument(
            id="doc-pending",
            owner_id="owner-one",
            filename="pending.pdf",
            storage_key="doc-pending.pdf",
            staging_key=".staging-pending.part",
            uploaded_at="2026-10-31T00:00:00+00:00",
            size_bytes=1,
            page_count=1,
            text_status="extracted",
            idempotency_key="pending-key",
            content_sha256="hash",
        ),
        [{"page_number": 1, "text": "page"}],
    )
    create_ready_document(repository, "doc-ready", "owner-two")

    repository.replace_document_embeddings(
        "owner-one",
        "doc-pending",
        "model-v1",
        [make_embedded_chunk("doc-pending")],
    )
    assert repository.get_owner_embeddings("owner-one")[0].document_id == "doc-pending"
    assert repository.get_owner_index_generation("owner-one") == 1
    with repository._connect() as connection:
        status = connection.execute(
            "SELECT status FROM documents WHERE id = 'doc-pending'"
        ).fetchone()["status"]
    assert status == "ready"

    with pytest.raises(DocumentNotReadyError):
        repository.replace_document_embeddings(
            "owner-one",
            "doc-ready",
            "model-v1",
            [make_embedded_chunk("doc-ready")],
        )
    assert len(repository.get_owner_embeddings("owner-one")) == 1
    assert repository.get_owner_embeddings("owner-two") == []


def test_owner_isolation_and_vector_id_resolution(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    create_ready_document(repository, "doc-one", "owner-one")
    create_ready_document(repository, "doc-two", "owner-two")
    repository.replace_document_embeddings(
        "owner-one", "doc-one", "model-v1", [make_embedded_chunk("doc-one")]
    )
    repository.replace_document_embeddings(
        "owner-two", "doc-two", "model-v1", [make_embedded_chunk("doc-two")]
    )
    owner_one_record = repository.get_owner_embeddings("owner-one")[0]
    owner_two_record = repository.get_owner_embeddings("owner-two")[0]

    assert owner_one_record.owner_id == "owner-one"
    assert owner_one_record.document_name == "doc-one.pdf"
    assert repository.resolve_embedding_ids(
        "owner-one", [owner_one_record.vector_id, owner_two_record.vector_id]
    ) == [owner_one_record]
    assert repository.resolve_embedding_ids(
        "owner-two", [owner_one_record.vector_id]
    ) == []


def test_reads_exclude_vectors_for_documents_that_are_not_ready(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    create_ready_document(repository, "doc-one", "owner-one")
    repository.replace_document_embeddings(
        "owner-one", "doc-one", "model-v1", [make_embedded_chunk("doc-one")]
    )
    vector_id = repository.get_owner_embeddings("owner-one")[0].vector_id

    with repository._connect() as connection:
        connection.execute(
            "UPDATE documents SET status = 'pending' WHERE id = 'doc-one'"
        )

    assert repository.get_owner_embeddings("owner-one") == []
    assert repository.resolve_embedding_ids("owner-one", [vector_id]) == []


def test_rolls_back_vector_replacement_and_generation_on_insert_failure(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    create_ready_document(repository, "doc-one", "owner-one")
    repository.replace_document_embeddings(
        "owner-one", "doc-one", "model-v1", [make_embedded_chunk("doc-one")]
    )
    original_records = repository.get_owner_embeddings("owner-one")
    original_generation = repository.get_owner_index_generation("owner-one")

    with repository._connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_second_chunk
            BEFORE INSERT ON embedded_chunks
            WHEN NEW.chunk_index = 1
            BEGIN
                SELECT RAISE(ABORT, 'injected insert failure');
            END
            """
        )

    with pytest.raises(RepositoryOperationError):
        repository.replace_document_embeddings(
            "owner-one",
            "doc-one",
            "model-v2",
            [
                make_embedded_chunk("doc-one", chunk_index=0),
                make_embedded_chunk("doc-one", chunk_index=1),
            ],
        )

    assert repository.get_owner_embeddings("owner-one") == original_records
    assert repository.get_owner_index_generation("owner-one") == original_generation


def test_replacement_uses_new_vector_ids_and_increments_generation(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    create_ready_document(repository, "doc-one", "owner-one")
    first_generation = repository.replace_document_embeddings(
        "owner-one",
        "doc-one",
        "model-v1",
        [make_embedded_chunk("doc-one"), make_embedded_chunk("doc-one", 1)],
    )
    old_ids = [
        record.vector_id for record in repository.get_owner_embeddings("owner-one")
    ]

    second_generation = repository.replace_document_embeddings(
        "owner-one",
        "doc-one",
        "model-v2",
        [make_embedded_chunk("doc-one", chunk_index=0, chunk_id="replacement")],
    )
    current_records = repository.get_owner_embeddings("owner-one")

    assert first_generation == 1
    assert second_generation == 2
    assert repository.get_owner_index_generation("owner-one") == 2
    assert len(current_records) == 1
    assert current_records[0].chunk_id == "replacement"
    assert current_records[0].model_name == "model-v2"
    assert current_records[0].vector_id not in old_ids
    assert repository.resolve_embedding_ids("owner-one", old_ids) == []


def test_remove_document_embeddings_increments_generation(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    create_ready_document(repository, "doc-one", "owner-one")
    repository.replace_document_embeddings(
        "owner-one", "doc-one", "model-v1", [make_embedded_chunk("doc-one")]
    )

    generation = repository.remove_document_embeddings("owner-one", "doc-one")

    assert generation == 2
    assert repository.get_owner_index_generation("owner-one") == 2
    assert repository.get_owner_embeddings("owner-one") == []


def test_empty_replacement_clears_document_embeddings_and_advances_generation(
    tmp_path: Path,
) -> None:
    repository = make_repository(tmp_path)
    create_ready_document(repository, "doc-one", "owner-one")
    repository.replace_document_embeddings(
        "owner-one", "doc-one", "model-v1", [make_embedded_chunk("doc-one")]
    )

    generation = repository.replace_document_embeddings(
        "owner-one", "doc-one", "model-v1", []
    )

    assert generation == 2
    assert repository.get_owner_embeddings("owner-one") == []


def test_page_metadata_is_preserved_in_stored_record(tmp_path: Path) -> None:
    repository = make_repository(tmp_path)
    create_ready_document(repository, "doc-one", "owner-one")
    repository.replace_document_embeddings(
        "owner-one", "doc-one", "model-v1", [make_embedded_chunk("doc-one")]
    )

    record = repository.get_owner_embeddings("owner-one")[0]

    assert record.page_numbers == (2, 3)
    assert record.start_page == 2
    assert record.end_page == 3
