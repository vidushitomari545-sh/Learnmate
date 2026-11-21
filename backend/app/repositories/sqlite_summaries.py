import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.repositories.interfaces import RepositoryOperationError
from app.repositories.sqlite_documents import SQLiteDocumentRepository
from app.repositories.summary_interfaces import (
    SummaryDocumentRecord,
    SummaryJobPhase,
    SummaryJobRecord,
)

_SUMMARY_ERROR_CODES = {
    "document_changed",
    "document_unavailable",
    "interrupted",
    "malformed_output",
    "no_extractable_text",
    "provider_authentication",
    "provider_failure",
    "provider_rate_limit",
    "provider_timeout",
    "provider_transient",
    "summary_context_limit",
    "summary_failed",
}


class SQLiteSummaryRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._documents = SQLiteDocumentRepository(database_path)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _document_record(row: sqlite3.Row) -> SummaryDocumentRecord:
        return SummaryDocumentRecord(
            id=row["id"],
            owner_id=row["owner_id"],
            filename=row["filename"],
            content_sha256=row["content_sha256"],
            page_count=row["page_count"] if row["page_count"] is not None else 0,
            text_status=row["text_status"] or "no_text_found",
        )

    @staticmethod
    def _read_result(value: str | None) -> dict[str, object] | None:
        if value is None:
            return None
        try:
            result = json.loads(value)
        except (json.JSONDecodeError, TypeError) as error:
            raise RepositoryOperationError(
                "Stored document summary is invalid."
            ) from error
        if not isinstance(result, dict):
            raise RepositoryOperationError("Stored document summary is invalid.")
        return result

    @classmethod
    def _job_record(cls, row: sqlite3.Row) -> SummaryJobRecord:
        status = row["status"]
        phase = row["phase"]
        if status not in {"queued", "processing", "completed", "failed"} or phase not in {
            "queued",
            "mapping",
            "reducing",
            "finalizing",
            "completed",
            "failed",
        }:
            raise RepositoryOperationError("Stored summary job state is invalid.")
        return SummaryJobRecord(
            id=row["id"],
            document_id=row["document_id"],
            owner_id=row["owner_id"],
            content_sha256=row["content_sha256"],
            summary_version=row["summary_version"],
            filename=row["filename"],
            page_count=row["page_count"],
            text_status=row["text_status"],
            status=status,
            phase=phase,
            progress_current=row["progress_current"],
            progress_total=row["progress_total"],
            result=cls._read_result(row["result_json"]),
            error_code=row["error_code"],
            retryable=bool(row["retryable"]),
            cache_hit=bool(row["cache_hit"]),
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def get_summary_document(
        self, owner_id: str, document_id: str
    ) -> SummaryDocumentRecord | None:
        self._documents.initialize()
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT id, owner_id, filename, content_sha256,
                           page_count, text_status
                    FROM documents
                    WHERE id = ? AND owner_id = ? AND status = 'ready'
                    """,
                    (document_id, owner_id),
                ).fetchone()
            return self._document_record(row) if row is not None else None
        except (sqlite3.Error, OSError) as error:
            raise RepositoryOperationError(
                "Summary document could not be loaded."
            ) from error

    def create_summary_job(
        self, owner_id: str, document_id: str, summary_version: int
    ) -> tuple[SummaryDocumentRecord, SummaryJobRecord, bool] | None:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must not be empty.")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must not be empty.")
        if (
            isinstance(summary_version, bool)
            or not isinstance(summary_version, int)
            or summary_version < 1
        ):
            raise ValueError("summary_version must be a positive integer.")

        self._documents.initialize()
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                document_row = connection.execute(
                    """
                    SELECT id, owner_id, filename, content_sha256,
                           page_count, text_status
                    FROM documents
                    WHERE id = ? AND owner_id = ? AND status = 'ready'
                    """,
                    (document_id, owner_id),
                ).fetchone()
                if document_row is None:
                    connection.rollback()
                    return None
                document = self._document_record(document_row)

                active_row = connection.execute(
                    """
                    SELECT * FROM document_summary_jobs
                    WHERE document_id = ? AND owner_id = ?
                      AND status IN ('queued', 'processing')
                    ORDER BY created_at DESC LIMIT 1
                    """,
                    (document_id, owner_id),
                ).fetchone()
                if active_row is not None:
                    if active_row["content_sha256"] == document.content_sha256:
                        connection.commit()
                        return document, self._job_record(active_row), False
                    connection.execute(
                        """
                        UPDATE document_summary_jobs
                        SET status = 'failed', phase = 'failed',
                            error_code = 'document_changed', retryable = 1,
                            updated_at = ?
                        WHERE id = ? AND status IN ('queued', 'processing')
                        """,
                        (self._now(), active_row["id"]),
                    )

                now = self._now()
                job_id = str(uuid.uuid4())
                cached_row = None
                if document.content_sha256 is not None:
                    cached_row = connection.execute(
                        """
                        SELECT result_json FROM document_summary_cache
                        WHERE document_id = ? AND owner_id = ?
                          AND content_sha256 = ? AND summary_version = ?
                        """,
                        (
                            document.id,
                            document.owner_id,
                            document.content_sha256,
                            summary_version,
                        ),
                    ).fetchone()

                if cached_row is not None:
                    result = self._read_result(cached_row["result_json"])
                    if result is None:
                        raise RepositoryOperationError(
                            "Stored document summary is invalid."
                        )
                    chunk_count = result.get("chunk_count", 0)
                    if (
                        isinstance(chunk_count, bool)
                        or not isinstance(chunk_count, int)
                        or chunk_count < 0
                    ):
                        raise RepositoryOperationError(
                            "Stored document summary progress is invalid."
                        )
                    connection.execute(
                        """
                        INSERT INTO document_summary_jobs (
                            id, document_id, owner_id, content_sha256,
                            summary_version, filename, page_count, text_status,
                            status, phase, progress_current, progress_total,
                            result_json, cache_hit, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'completed', 'completed',
                                  ?, ?, ?, 1, ?, ?)
                        """,
                        (
                            job_id,
                            document.id,
                            document.owner_id,
                            document.content_sha256,
                            summary_version,
                            document.filename,
                            document.page_count,
                            document.text_status,
                            chunk_count,
                            chunk_count,
                            cached_row["result_json"],
                            now,
                            now,
                        ),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO document_summary_jobs (
                            id, document_id, owner_id, content_sha256,
                            summary_version, filename, page_count, text_status,
                            status, phase, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'queued', 'queued', ?, ?)
                        """,
                        (
                            job_id,
                            document.id,
                            document.owner_id,
                            document.content_sha256,
                            summary_version,
                            document.filename,
                            document.page_count,
                            document.text_status,
                            now,
                            now,
                        ),
                    )
                row = connection.execute(
                    "SELECT * FROM document_summary_jobs WHERE id = ?", (job_id,)
                ).fetchone()
                connection.commit()
                if row is None:
                    raise RepositoryOperationError(
                        "Summary job could not be created."
                    )
                return document, self._job_record(row), cached_row is None
        except (sqlite3.Error, OSError) as error:
            raise RepositoryOperationError(
                "Summary job could not be created."
            ) from error

    def get_summary_job(
        self, owner_id: str, document_id: str, job_id: str
    ) -> SummaryJobRecord | None:
        self._documents.initialize()
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT jobs.* FROM document_summary_jobs AS jobs
                    JOIN documents
                      ON documents.id = jobs.document_id
                     AND documents.owner_id = jobs.owner_id
                    WHERE jobs.id = ? AND jobs.document_id = ?
                      AND jobs.owner_id = ? AND documents.status = 'ready'
                      AND (
                        jobs.content_sha256 IS NULL
                        OR jobs.content_sha256 = documents.content_sha256
                      )
                    """,
                    (job_id, document_id, owner_id),
                ).fetchone()
            return self._job_record(row) if row is not None else None
        except (sqlite3.Error, OSError) as error:
            raise RepositoryOperationError(
                "Summary job could not be loaded."
            ) from error

    def claim_summary_job(self, job_id: str) -> bool:
        self._documents.initialize()
        try:
            with closing(self._connect()) as connection, connection:
                cursor = connection.execute(
                    """
                    UPDATE document_summary_jobs
                    SET status = 'processing', phase = 'mapping',
                        progress_current = 0, progress_total = 0, updated_at = ?
                    WHERE id = ? AND status = 'queued'
                    """,
                    (self._now(), job_id),
                )
                return cursor.rowcount == 1
        except (sqlite3.Error, OSError) as error:
            raise RepositoryOperationError(
                "Summary job could not be claimed."
            ) from error

    def update_summary_progress(
        self,
        job_id: str,
        phase: SummaryJobPhase,
        current: int,
        total: int,
    ) -> None:
        if phase not in {"mapping", "reducing", "finalizing"}:
            raise ValueError("Summary phase is invalid.")
        if (
            isinstance(current, bool)
            or not isinstance(current, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or current < 0
            or total < 0
            or current > total
        ):
            raise ValueError("Summary progress must be a valid bounded count.")
        self._documents.initialize()
        try:
            with closing(self._connect()) as connection, connection:
                cursor = connection.execute(
                    """
                    UPDATE document_summary_jobs
                    SET phase = ?, progress_current = ?, progress_total = ?,
                        updated_at = ?
                    WHERE id = ? AND status = 'processing'
                    """,
                    (phase, current, total, self._now(), job_id),
                )
                if cursor.rowcount != 1:
                    raise RepositoryOperationError(
                        "Summary job is no longer processing."
                    )
        except (sqlite3.Error, OSError) as error:
            raise RepositoryOperationError(
                "Summary progress could not be saved."
            ) from error

    def complete_summary_job(
        self, job_id: str, result: dict[str, object]
    ) -> bool:
        if not isinstance(result, dict):
            raise ValueError("Summary result must be a mapping.")
        result_json = json.dumps(
            result, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        self._documents.initialize()
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                job = connection.execute(
                    "SELECT * FROM document_summary_jobs WHERE id = ?",
                    (job_id,),
                ).fetchone()
                if job is None or job["status"] != "processing":
                    connection.rollback()
                    return False
                document = connection.execute(
                    """
                    SELECT status, content_sha256 FROM documents
                    WHERE id = ? AND owner_id = ?
                    """,
                    (job["document_id"], job["owner_id"]),
                ).fetchone()
                if (
                    document is None
                    or document["status"] != "ready"
                    or document["content_sha256"] != job["content_sha256"]
                ):
                    connection.execute(
                        """
                        UPDATE document_summary_jobs
                        SET status = 'failed', phase = 'failed',
                            error_code = 'document_changed', retryable = 1,
                            updated_at = ?
                        WHERE id = ? AND status = 'processing'
                        """,
                        (self._now(), job_id),
                    )
                    connection.commit()
                    return False

                now = self._now()
                chunk_count = result.get("chunk_count")
                if (
                    isinstance(chunk_count, bool)
                    or not isinstance(chunk_count, int)
                    or chunk_count < 1
                ):
                    raise ValueError("Summary result has invalid chunk_count.")
                if job["content_sha256"] is not None:
                    connection.execute(
                        """
                        INSERT OR REPLACE INTO document_summary_cache (
                            document_id, owner_id, content_sha256, summary_version,
                            result_json, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            job["document_id"],
                            job["owner_id"],
                            job["content_sha256"],
                            job["summary_version"],
                            result_json,
                            now,
                        ),
                    )
                connection.execute(
                    """
                    UPDATE document_summary_jobs
                    SET status = 'completed', phase = 'completed',
                        progress_current = ?, progress_total = ?,
                        result_json = ?, error_code = NULL, retryable = 0,
                        updated_at = ?
                    WHERE id = ? AND status = 'processing'
                    """,
                    (chunk_count, chunk_count, result_json, now, job_id),
                )
                connection.commit()
                return True
        except (sqlite3.Error, OSError, json.JSONDecodeError) as error:
            raise RepositoryOperationError(
                "Summary result could not be saved."
            ) from error

    def fail_summary_job(
        self, job_id: str, error_code: str, retryable: bool
    ) -> None:
        if error_code not in _SUMMARY_ERROR_CODES:
            raise ValueError("Summary error code is invalid.")
        self._documents.initialize()
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    UPDATE document_summary_jobs
                    SET status = 'failed', phase = 'failed', error_code = ?,
                        retryable = ?, updated_at = ?
                    WHERE id = ? AND status IN ('queued', 'processing')
                    """,
                    (error_code, int(retryable), self._now(), job_id),
                )
        except (sqlite3.Error, OSError) as error:
            raise RepositoryOperationError(
                "Summary failure could not be saved."
            ) from error

    def recover_summary_jobs(self) -> list[SummaryJobRecord]:
        self._documents.initialize()
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                now = self._now()
                connection.execute(
                    """
                    UPDATE document_summary_jobs
                    SET status = 'failed', phase = 'failed',
                        error_code = 'interrupted', retryable = 1, updated_at = ?
                    WHERE status = 'processing'
                    """,
                    (now,),
                )
                rows = connection.execute(
                    """
                    SELECT * FROM document_summary_jobs
                    WHERE status = 'queued'
                    ORDER BY created_at
                    """
                ).fetchall()
                connection.commit()
            return [self._job_record(row) for row in rows]
        except (sqlite3.Error, OSError) as error:
            raise RepositoryOperationError(
                "Pending summary jobs could not be recovered."
            ) from error
