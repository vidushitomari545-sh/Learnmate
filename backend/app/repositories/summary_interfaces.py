from dataclasses import dataclass
from typing import Literal, Protocol

SummaryJobStatus = Literal["queued", "processing", "completed", "failed"]
SummaryJobPhase = Literal[
    "queued", "mapping", "reducing", "finalizing", "completed", "failed"
]


@dataclass(frozen=True)
class SummaryDocumentRecord:
    id: str
    owner_id: str
    filename: str
    content_sha256: str | None
    page_count: int
    text_status: str


@dataclass(frozen=True)
class SummaryJobRecord:
    id: str
    document_id: str
    owner_id: str
    content_sha256: str | None
    summary_version: int
    filename: str
    page_count: int
    text_status: str
    status: SummaryJobStatus
    phase: SummaryJobPhase
    progress_current: int
    progress_total: int
    result: dict[str, object] | None
    error_code: str | None
    retryable: bool
    cache_hit: bool


class SummaryRepository(Protocol):
    def get_summary_document(
        self, owner_id: str, document_id: str
    ) -> SummaryDocumentRecord | None: ...

    def create_summary_job(
        self, owner_id: str, document_id: str, summary_version: int
    ) -> tuple[SummaryDocumentRecord, SummaryJobRecord, bool] | None: ...

    def get_summary_job(
        self, owner_id: str, document_id: str, job_id: str
    ) -> SummaryJobRecord | None: ...

    def claim_summary_job(self, job_id: str) -> bool: ...

    def update_summary_progress(
        self,
        job_id: str,
        phase: SummaryJobPhase,
        current: int,
        total: int,
    ) -> None: ...

    def complete_summary_job(
        self, job_id: str, result: dict[str, object]
    ) -> bool: ...

    def fail_summary_job(
        self, job_id: str, error_code: str, retryable: bool
    ) -> None: ...

    def recover_summary_jobs(self) -> list[SummaryJobRecord]: ...
