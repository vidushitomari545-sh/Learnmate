from dataclasses import dataclass
from typing import Protocol

from app.services.embeddings.types import EmbeddedChunk


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    owner_id: str | None
    filename: str
    storage_key: str
    uploaded_at: str
    size_bytes: int
    status: str
    page_count: int | None
    text_status: str | None
    idempotency_key: str | None
    content_sha256: str | None
    staging_key: str | None

    def summary(self) -> dict[str, str | int | None]:
        return {
            "id": self.id,
            "filename": self.filename,
            "uploaded_at": self.uploaded_at,
            "size_bytes": self.size_bytes,
            "page_count": self.page_count,
            "text_status": self.text_status,
        }


@dataclass(frozen=True)
class NewDocument:
    id: str
    owner_id: str
    filename: str
    storage_key: str
    staging_key: str
    uploaded_at: str
    size_bytes: int
    page_count: int
    text_status: str
    idempotency_key: str
    content_sha256: str


class DuplicateIdempotencyKeyError(Exception):
    """Raised when another request already owns an idempotency key."""


class RepositoryOperationError(RuntimeError):
    """Raised when a repository operation fails."""


class InvalidDocumentCursorError(ValueError):
    """Raised when a document-list cursor is invalid."""


class DocumentNotReadyError(LookupError):
    """Raised when an owner cannot access a ready document."""


@dataclass(frozen=True)
class StoredEmbeddingRecord:
    vector_id: int
    chunk_id: str
    document_id: str
    document_name: str
    owner_id: str
    chunk_index: int
    text: str
    page_numbers: tuple[int, ...]
    start_page: int
    end_page: int
    embedding: tuple[float, ...]
    dimension: int
    encoding: str
    model_name: str


class DocumentRepository(Protocol):
    def initialize(self) -> None: ...

    def find_by_idempotency_key(
        self, owner_id: str, idempotency_key: str
    ) -> DocumentRecord | None: ...

    def create_pending(
        self, document: NewDocument, pages: list[dict[str, str | int]]
    ) -> None: ...

    def mark_ready(self, owner_id: str, document_id: str) -> None: ...

    def delete_pending(self, owner_id: str, document_id: str) -> None: ...

    def list_pending(self) -> list[DocumentRecord]: ...

    def list_documents(
        self, owner_id: str, limit: int, cursor: str | None
    ) -> tuple[list[dict[str, str | int | None]], str | None, int]: ...

    def get_document_pages(
        self, owner_id: str, document_id: str
    ) -> list[dict[str, str | int]] | None: ...

    def get_document_pages_for_indexing(
        self, owner_id: str, document_id: str
    ) -> list[dict[str, str | int]] | None: ...

    def replace_document_embeddings(
        self,
        owner_id: str,
        document_id: str,
        model_name: str,
        chunks: list[EmbeddedChunk],
    ) -> int: ...

    def get_owner_embeddings(self, owner_id: str) -> list[StoredEmbeddingRecord]: ...

    def resolve_embedding_ids(
        self, owner_id: str, vector_ids: list[int]
    ) -> list[StoredEmbeddingRecord]: ...

    def remove_document_embeddings(self, owner_id: str, document_id: str) -> int: ...

    def get_owner_index_generation(self, owner_id: str) -> int: ...
