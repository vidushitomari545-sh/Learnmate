import logging
import uuid
from datetime import datetime, timezone
from typing import BinaryIO

from app.documents import (
    DocumentTooLargeError,
    EmptyFileError,
    InvalidPDFError,
    PDFTextExtractionError,
    extract_pdf_text,
    validate_pdf_upload,
)
from app.repositories.interfaces import (
    DocumentRecord,
    DocumentRepository,
    DuplicateIdempotencyKeyError,
    InvalidDocumentCursorError,
    NewDocument,
    RepositoryOperationError,
)
from app.services.embeddings.service import EmbeddingService
from app.storage.interfaces import (
    DocumentStorage,
    StagedFile,
    StorageOperationError,
)
from app.services.text_chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    TextChunk,
    chunk_text_pages,
)

logger = logging.getLogger(__name__)


class IdempotencyConflictError(ValueError):
    """Raised when an idempotency key is reused for different content."""


class UploadInProgressError(RuntimeError):
    """Raised when an idempotent upload has a pending concurrent attempt."""


class DocumentPersistenceError(RuntimeError):
    """Raised when a document could not be safely persisted."""


class DocumentService:
    def __init__(
        self,
        repository: DocumentRepository,
        storage: DocumentStorage,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.repository = repository
        self.storage = storage
        self.embedding_service = embedding_service

    def _index_pending_document(self, owner_id: str, document_id: str) -> None:
        if self.embedding_service is None:
            raise DocumentPersistenceError(
                "Document indexing is not configured."
            )
        chunks = self._chunk_pending_document(owner_id, document_id)
        if chunks is None:
            raise DocumentPersistenceError(
                "The pending document pages could not be loaded."
            )
        embedded_chunks = self.embedding_service.embed_chunks(chunks)
        self.repository.replace_document_embeddings(
            owner_id,
            document_id,
            self.embedding_service.model_identifier,
            embedded_chunks,
        )

    @staticmethod
    def _staged_file(record: DocumentRecord) -> StagedFile:
        if record.staging_key is None or record.content_sha256 is None:
            raise DocumentPersistenceError("Pending upload metadata is incomplete.")
        return StagedFile(
            storage_key=record.storage_key,
            staging_key=record.staging_key,
            size_bytes=record.size_bytes,
            content_sha256=record.content_sha256,
        )

    def _recover_record(self, record: DocumentRecord) -> DocumentRecord | None:
        if record.status == "ready":
            return record
        if record.owner_id is None or record.idempotency_key is None:
            raise DocumentPersistenceError(
                "Pending upload ownership metadata is incomplete."
            )
        staged = self._staged_file(record)
        if not self.storage.recover(staged):
            self.repository.delete_pending(record.owner_id, record.id)
            return None
        self._index_pending_document(record.owner_id, record.id)
        return self.repository.find_by_idempotency_key(
            record.owner_id, record.idempotency_key
        )

    def recover_pending_uploads(self) -> None:
        for record in self.repository.list_pending():
            try:
                recovered = self._recover_record(record)
                if recovered is None:
                    logger.warning(
                        "Removed incomplete pending upload %s with no staged file.",
                        record.id,
                    )
            except Exception:
                logger.exception("Could not recover pending upload %s.", record.id)

    def upload(
        self,
        owner_id: str,
        idempotency_key: str,
        filename: str | None,
        content_type: str | None,
        file: BinaryIO,
    ) -> dict[str, str | int | None]:
        validated = validate_pdf_upload(filename, content_type, file)
        extracted = extract_pdf_text(validated.reader)
        document_id = str(uuid.uuid4())
        try:
            staged = self.storage.stage(file, document_id)
        except StorageOperationError as error:
            raise DocumentPersistenceError(
                "The uploaded document could not be staged."
            ) from error
        pending_created = False

        if staged.size_bytes != validated.size_bytes:
            self._cleanup_staged(staged)
            raise DocumentPersistenceError("Uploaded file size changed during processing.")

        try:
            existing = self.repository.find_by_idempotency_key(
                owner_id, idempotency_key
            )
            if existing is not None:
                if existing.content_sha256 != staged.content_sha256:
                    raise IdempotencyConflictError(
                        "This idempotency key was already used for different content."
                    )
                if existing.status == "pending":
                    self._cleanup_staged(staged)
                    raise UploadInProgressError(
                        "This upload is already in progress. Retry shortly."
                    )
                self._cleanup_staged(staged)
                return self._response(existing)

            document = NewDocument(
                id=document_id,
                owner_id=owner_id,
                filename=filename or "",
                storage_key=staged.storage_key,
                staging_key=staged.staging_key,
                uploaded_at=datetime.now(timezone.utc).isoformat(),
                size_bytes=validated.size_bytes,
                page_count=extracted["page_count"],
                text_status=extracted["text_status"],
                idempotency_key=idempotency_key,
                content_sha256=staged.content_sha256,
            )
            try:
                self.repository.create_pending(document, extracted["pages"])
                pending_created = True
            except DuplicateIdempotencyKeyError:
                existing = self.repository.find_by_idempotency_key(
                    owner_id, idempotency_key
                )
                if existing is None:
                    raise
                if existing.content_sha256 != staged.content_sha256:
                    raise IdempotencyConflictError(
                        "This idempotency key was already used for different content."
                    )
                if existing.status == "pending":
                    self._cleanup_staged(staged)
                    raise UploadInProgressError(
                        "This upload is already in progress. Retry shortly."
                    )
                self._cleanup_staged(staged)
                return self._response(existing)

            self.storage.promote(staged)
            self._index_pending_document(owner_id, document_id)
            pending_created = False
            return self._upload_response(document)
        except (
            IdempotencyConflictError,
            UploadInProgressError,
            DocumentPersistenceError,
        ):
            if pending_created:
                self._rollback_pending(owner_id, document_id)
            self._cleanup_staged(staged)
            raise
        except (StorageOperationError, RepositoryOperationError) as error:
            if pending_created:
                self._rollback_pending(owner_id, document_id)
            self._cleanup_staged(staged)
            raise DocumentPersistenceError(
                "The uploaded document could not be saved."
            ) from error
        except DuplicateIdempotencyKeyError as error:
            if pending_created:
                self._rollback_pending(owner_id, document_id)
            self._cleanup_staged(staged)
            raise DocumentPersistenceError(
                "The upload request conflicted with another attempt."
            ) from error
        except Exception as error:
            if pending_created:
                self._rollback_pending(owner_id, document_id)
            self._cleanup_staged(staged)
            logger.exception("Could not index uploaded document %s.", document_id)
            raise DocumentPersistenceError(
                "The uploaded document could not be indexed."
            ) from error

    def _rollback_pending(self, owner_id: str, document_id: str) -> None:
        try:
            self.repository.delete_pending(owner_id, document_id)
        except RepositoryOperationError:
            logger.exception(
                "Could not remove pending document record %s.", document_id
            )

    def _cleanup_staged(self, staged: StagedFile) -> None:
        try:
            self.storage.cleanup(staged)
        except StorageOperationError:
            logger.exception(
                "Could not clean up temporary upload %s.", staged.staging_key
            )

    @staticmethod
    def _response(record: DocumentRecord) -> dict[str, str | int | None]:
        response = record.summary()
        response.update(
            {
                "status": "stored",
                "text_status": record.text_status,
                "page_count": record.page_count,
            }
        )
        return response

    @staticmethod
    def _upload_response(document: NewDocument) -> dict[str, str | int | None]:
        return {
            "id": document.id,
            "filename": document.filename,
            "uploaded_at": document.uploaded_at,
            "size_bytes": document.size_bytes,
            "status": "stored",
            "page_count": document.page_count,
            "text_status": document.text_status,
        }

    def list_documents(
        self, owner_id: str, limit: int, cursor: str | None
    ) -> tuple[list[dict[str, str | int | None]], str | None, int]:
        return self.repository.list_documents(owner_id, limit, cursor)

    def get_document_pages(
        self, owner_id: str, document_id: str
    ) -> list[dict[str, str | int]] | None:
        return self.repository.get_document_pages(owner_id, document_id)

    def chunk_document(
        self,
        owner_id: str,
        document_id: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[TextChunk] | None:
        pages = self.repository.get_document_pages(owner_id, document_id)
        if pages is None:
            return None
        return chunk_text_pages(document_id, pages, chunk_size, overlap)

    def _chunk_pending_document(
        self,
        owner_id: str,
        document_id: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[TextChunk] | None:
        pages = self.repository.get_document_pages_for_indexing(
            owner_id, document_id
        )
        if pages is None:
            return None
        return chunk_text_pages(document_id, pages, chunk_size, overlap)
