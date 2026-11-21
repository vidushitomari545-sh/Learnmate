import logging

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile

from app.documents import (
    DocumentTooLargeError,
    EmptyFileError,
    InvalidPDFError,
    PDFTextExtractionError,
)
from app.schemas.documents import (
    DocumentListResponse,
    DocumentPagesResponse,
    DocumentSummaryJobResponse,
    StoredUploadResponse,
)
from app.repositories.interfaces import (
    InvalidDocumentCursorError,
    RepositoryOperationError,
)
from app.services.dependencies import (
    get_development_owner_id,
    get_document_service,
    get_document_summary_job_service,
)
from app.services.document_service import (
    DocumentPersistenceError,
    DocumentService,
    IdempotencyConflictError,
    UploadInProgressError,
)
from app.services.document_summary_jobs import DocumentSummaryJobService
from app.repositories.summary_interfaces import SummaryJobRecord

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


def _summary_job_response(job: SummaryJobRecord) -> dict[str, object]:
    if job.status == "completed" and job.result is None:
        raise RepositoryOperationError("Completed summary has no result.")
    return {
        "id": job.id,
        "document_id": job.document_id,
        "filename": job.filename,
        "page_count": job.page_count,
        "text_status": job.text_status,
        "status": job.status,
        "phase": job.phase,
        "progress_current": job.progress_current,
        "progress_total": job.progress_total,
        "cache_hit": job.cache_hit,
        "retryable": job.retryable,
        "error_code": job.error_code,
        "result": job.result if job.status == "completed" else None,
    }


@router.post("/upload", response_model=StoredUploadResponse)
def upload_document(
    file: UploadFile = File(...),
    idempotency_key: str = Header(
        ..., alias="Idempotency-Key", min_length=1, max_length=128
    ),
    owner_id: str = Depends(get_development_owner_id),
    service: DocumentService = Depends(get_document_service),
) -> dict[str, str | int | None]:
    idempotency_key = idempotency_key.strip()
    if not idempotency_key:
        raise HTTPException(
            status_code=400, detail="Idempotency-Key must not be empty."
        )
    try:
        return service.upload(
            owner_id,
            idempotency_key.strip(),
            file.filename,
            file.content_type,
            file.file,
        )
    except InvalidPDFError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    except EmptyFileError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except DocumentTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except PDFTextExtractionError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    except IdempotencyConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except UploadInProgressError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (DocumentPersistenceError, RepositoryOperationError) as error:
        logger.exception("Document upload persistence failed.")
        raise HTTPException(
            status_code=500, detail="The uploaded document could not be saved."
        ) from error


@router.get("", response_model=DocumentListResponse)
def get_documents(
    limit: int = Query(default=20, ge=1, le=100),
    cursor: str | None = None,
    owner_id: str = Depends(get_development_owner_id),
    service: DocumentService = Depends(get_document_service),
) -> dict[str, object]:
    try:
        documents, next_cursor, total_count = service.list_documents(
            owner_id, limit, cursor
        )
    except InvalidDocumentCursorError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RepositoryOperationError as error:
        logger.exception("Document listing failed.")
        raise HTTPException(
            status_code=500, detail="Documents could not be loaded."
        ) from error
    return {
        "documents": documents,
        "next_cursor": next_cursor,
        "total_count": total_count,
    }


@router.get("/{document_id}/pages", response_model=DocumentPagesResponse)
def get_document_pages(
    document_id: str,
    owner_id: str = Depends(get_development_owner_id),
    service: DocumentService = Depends(get_document_service),
) -> dict[str, object]:
    try:
        pages = service.get_document_pages(owner_id, document_id)
    except RepositoryOperationError as error:
        logger.exception("Document page retrieval failed.")
        raise HTTPException(
            status_code=500, detail="Document pages could not be loaded."
        ) from error
    if pages is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"document_id": document_id, "pages": pages}


@router.post(
    "/{document_id}/summary",
    response_model=DocumentSummaryJobResponse,
    status_code=202,
)
def start_document_summary(
    document_id: str,
    owner_id: str = Depends(get_development_owner_id),
    service: DocumentSummaryJobService = Depends(
        get_document_summary_job_service
    ),
) -> dict[str, object]:
    try:
        job = service.start_summary(owner_id, document_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        return _summary_job_response(job)
    except RepositoryOperationError as error:
        logger.exception("Document summary could not be started.")
        raise HTTPException(
            status_code=500, detail="Document summary could not be started."
        ) from error


@router.get(
    "/{document_id}/summary/{job_id}",
    response_model=DocumentSummaryJobResponse,
)
def get_document_summary(
    document_id: str,
    job_id: str,
    owner_id: str = Depends(get_development_owner_id),
    service: DocumentSummaryJobService = Depends(
        get_document_summary_job_service
    ),
) -> dict[str, object]:
    try:
        job = service.get_summary_job(owner_id, document_id, job_id)
    except RepositoryOperationError as error:
        logger.exception("Document summary status could not be loaded.")
        raise HTTPException(
            status_code=500, detail="Document summary status could not be loaded."
        ) from error
    if job is None:
        raise HTTPException(status_code=404, detail="Document summary not found.")
    return _summary_job_response(job)
