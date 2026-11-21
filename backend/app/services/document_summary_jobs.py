"""Persistent background job coordination for document summaries."""

import logging
from concurrent.futures import Executor, ThreadPoolExecutor

from app.repositories.summary_interfaces import (
    SummaryJobRecord,
    SummaryRepository,
)
from app.services.document_service import DocumentService
from app.services.document_summary import (
    DocumentSummaryError,
    DocumentSummaryService,
    InvalidSummaryInputError,
    MalformedSummaryOutputError,
    NoExtractableTextError,
    SummaryResponseDiagnostic,
    SummaryContextLimitError,
)
from app.services.llm.interfaces import (
    LLMAuthenticationError,
    LLMMalformedResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTransientError,
)

logger = logging.getLogger(__name__)


class DocumentSummaryJobService:
    def __init__(
        self,
        repository: SummaryRepository,
        document_service: DocumentService,
        summary_service: DocumentSummaryService,
        *,
        executor: Executor | None = None,
    ) -> None:
        self.repository = repository
        self.document_service = document_service
        self.summary_service = summary_service
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="learnmate-summary"
        )
        self._owns_executor = executor is None
        self._recovery_started = False

    def close(self) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=True)

    def start_summary(
        self, owner_id: str, document_id: str
    ) -> SummaryJobRecord | None:
        created = self.repository.create_summary_job(
            owner_id, document_id, self.summary_service.prompt_version
        )
        if created is None:
            return None
        _, job, is_new = created
        if is_new:
            self._schedule(job)
        return self.repository.get_summary_job(owner_id, document_id, job.id) or job

    def get_summary_job(
        self, owner_id: str, document_id: str, job_id: str
    ) -> SummaryJobRecord | None:
        return self.repository.get_summary_job(owner_id, document_id, job_id)

    def recover_pending_jobs(self) -> None:
        if self._recovery_started:
            return
        self._recovery_started = True
        for job in self.repository.recover_summary_jobs():
            self._schedule(job)

    def _schedule(self, job: SummaryJobRecord) -> None:
        try:
            self._executor.submit(
                self._run_job,
                job.id,
                job.owner_id,
                job.document_id,
            )
        except RuntimeError:
            logger.exception("Could not schedule document summary job %s.", job.id)
            self._fail(job.id, "summary_failed", True)

    @staticmethod
    def _provider_failure(error: LLMProviderError) -> tuple[str, bool]:
        if isinstance(error, LLMTimeoutError):
            return "provider_timeout", True
        if isinstance(error, LLMRateLimitError):
            return "provider_rate_limit", True
        if isinstance(error, LLMTransientError):
            return "provider_transient", True
        if isinstance(error, LLMAuthenticationError):
            return "provider_authentication", False
        if isinstance(error, LLMMalformedResponseError):
            return "malformed_output", True
        return "provider_failure", False

    def _fail(self, job_id: str, error_code: str, retryable: bool) -> None:
        try:
            self.repository.fail_summary_job(job_id, error_code, retryable)
        except Exception:
            logger.exception("Could not save failure state for summary job %s.", job_id)

    @staticmethod
    def _log_response_diagnostic(
        job_id: str,
        diagnostic: object | None,
    ) -> None:
        if not isinstance(diagnostic, SummaryResponseDiagnostic):
            return
        logger.warning(
            "Summary response diagnostic job_id=%s stage=%s category=%s "
            "expected_contract=%s actual_response_type=%s source_ids_rejected=%s",
            job_id,
            diagnostic.stage,
            diagnostic.category,
            diagnostic.expected_contract,
            diagnostic.actual_response_type,
            diagnostic.source_ids_rejected,
        )
        shape = diagnostic.source_ids_shape
        if shape is not None:
            logger.warning(
                "Summary source_ids shape job_id=%s container_type=%s "
                "item_count=%s non_string_member_count=%s "
                "empty_or_whitespace_string_member_count=%s "
                "duplicate_string_member_count=%s unknown_id_count=%s "
                "list_empty=%s normalization_ran=%s",
                job_id,
                shape.container_type,
                shape.item_count,
                shape.non_string_member_count,
                shape.empty_string_member_count,
                shape.duplicate_string_member_count,
                shape.unknown_id_count,
                shape.is_empty,
                shape.normalization_ran,
            )

    @staticmethod
    def _safe_exception_type(error: Exception) -> str:
        name = type(error).__name__
        if (
            not name.isascii()
            or not name.replace("_", "").isalnum()
            or len(name) > 80
        ):
            return "UnknownException"
        return name

    def _run_job(self, job_id: str, owner_id: str, document_id: str) -> None:
        try:
            if not self.repository.claim_summary_job(job_id):
                return
            job = self.repository.get_summary_job(owner_id, document_id, job_id)
            document = self.repository.get_summary_document(owner_id, document_id)
            if job is None or document is None:
                self._fail(job_id, "document_unavailable", False)
                return
            if job.content_sha256 != document.content_sha256:
                self._fail(job_id, "document_changed", True)
                return

            pages = self.document_service.get_document_pages(owner_id, document_id)
            if pages is None:
                self._fail(job_id, "document_unavailable", False)
                return

            result = self.summary_service.summarize_pages(
                document_id,
                pages,
                page_count=job.page_count,
                text_status=job.text_status,
                progress=lambda phase, current, total: self.repository.update_summary_progress(
                    job_id, phase, current, total
                ),
            )
            self.repository.complete_summary_job(
                job_id,
                {
                    "summary_markdown": result.summary_markdown,
                    "page_references": list(result.page_references),
                    "pages_without_text": list(result.pages_without_text),
                    "chunk_count": result.chunk_count,
                    "text_status": result.text_status,
                },
            )
        except NoExtractableTextError:
            self._fail(job_id, "no_extractable_text", False)
        except MalformedSummaryOutputError as error:
            self._log_response_diagnostic(job_id, error.diagnostic)
            self._fail(job_id, "malformed_output", True)
        except SummaryContextLimitError:
            self._fail(job_id, "summary_context_limit", False)
        except InvalidSummaryInputError:
            self._fail(job_id, "document_unavailable", False)
        except DocumentSummaryError as error:
            logger.error(
                "Document summary job failed job_id=%s exception_type=%s.",
                job_id,
                self._safe_exception_type(error),
            )
            self._fail(job_id, "summary_failed", False)
        except LLMProviderError as error:
            self._log_response_diagnostic(job_id, error.summary_diagnostic)
            error_code, retryable = self._provider_failure(error)
            logger.warning(
                "Summary provider request failed for job %s (%s).",
                job_id,
                error_code,
            )
            self._fail(job_id, error_code, retryable)
        except Exception as error:
            self._log_response_diagnostic(
                job_id, getattr(error, "summary_diagnostic", None)
            )
            logger.error(
                "Document summary job failed unexpectedly job_id=%s "
                "exception_type=%s.",
                job_id,
                self._safe_exception_type(error),
            )
            self._fail(job_id, "summary_failed", False)
