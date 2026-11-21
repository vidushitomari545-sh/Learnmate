from typing import Literal

from pydantic import BaseModel


class DocumentSummary(BaseModel):
    id: str
    filename: str
    uploaded_at: str
    size_bytes: int
    page_count: int | None
    text_status: Literal["text_found", "partial_text", "no_text_found"] | None


class StoredUploadResponse(DocumentSummary):
    status: Literal["stored"]


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    next_cursor: str | None
    total_count: int


class DocumentPage(BaseModel):
    page_number: int
    text: str


class DocumentPagesResponse(BaseModel):
    document_id: str
    pages: list[DocumentPage]


class DocumentSummaryResultResponse(BaseModel):
    summary_markdown: str
    page_references: list[int]
    pages_without_text: list[int]
    chunk_count: int
    text_status: Literal["text_found", "partial_text"]


class DocumentSummaryJobResponse(BaseModel):
    id: str
    document_id: str
    filename: str
    page_count: int
    text_status: Literal["text_found", "partial_text", "no_text_found"]
    status: Literal["queued", "processing", "completed", "failed"]
    phase: Literal[
        "queued", "mapping", "reducing", "finalizing", "completed", "failed"
    ]
    progress_current: int
    progress_total: int
    cache_hit: bool
    retryable: bool
    error_code: str | None
    result: DocumentSummaryResultResponse | None
