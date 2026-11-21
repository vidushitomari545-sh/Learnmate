import base64
import binascii
import functools
import json
import math
import sqlite3
import struct
from contextlib import closing
from pathlib import Path
from numbers import Real
from typing import Callable, TypeVar

from app.repositories.interfaces import (
    DocumentRecord,
    DocumentNotReadyError,
    DuplicateIdempotencyKeyError,
    InvalidDocumentCursorError,
    NewDocument,
    RepositoryOperationError,
    StoredEmbeddingRecord,
)
from app.services.embeddings.types import EmbeddedChunk

_T = TypeVar("_T")
EMBEDDING_DIMENSION = 384
EMBEDDING_ENCODING = "float32-le-v1"
_EMBEDDING_BLOB_SIZE = EMBEDDING_DIMENSION * 4


def _translate_sqlite_errors(
    operation: Callable[..., _T],
) -> Callable[..., _T]:
    @functools.wraps(operation)
    def wrapped(*args: object, **kwargs: object) -> _T:
        try:
            return operation(*args, **kwargs)
        except (sqlite3.Error, OSError) as error:
            raise RepositoryOperationError("Database operation failed.") from error

    return wrapped


_DOCUMENT_COLUMNS = (
    ("owner_id", "TEXT NULL"),
    ("status", "TEXT NOT NULL DEFAULT 'ready'"),
    ("page_count", "INTEGER NULL"),
    ("text_status", "TEXT NULL"),
    ("idempotency_key", "TEXT NULL"),
    ("content_sha256", "TEXT NULL"),
    ("staging_filename", "TEXT NULL"),
)


def _migration_baseline(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            stored_filename TEXT NOT NULL UNIQUE,
            uploaded_at TEXT NOT NULL,
            size_bytes INTEGER NOT NULL
        )
        """
    )


def _migration_document_metadata(connection: sqlite3.Connection) -> None:
    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(documents)")
    }
    for name, definition in _DOCUMENT_COLUMNS:
        if name not in columns:
            connection.execute(
                f"ALTER TABLE documents ADD COLUMN {name} {definition}"
            )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_owner_idempotency
        ON documents(owner_id, idempotency_key)
        WHERE idempotency_key IS NOT NULL
        """
    )


def _migration_document_pages(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS document_pages (
            document_id TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            text TEXT NOT NULL,
            PRIMARY KEY (document_id, page_number),
            FOREIGN KEY (document_id) REFERENCES documents(id)
                ON DELETE CASCADE
        )
        """
    )


def _migration_document_list_index(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_documents_owner_status_uploaded
        ON documents(owner_id, status, uploaded_at DESC, id DESC)
        """
    )


def _migration_embedded_chunks(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_id_owner "
        "ON documents(id, owner_id)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS owner_embedding_generations (
            owner_id TEXT PRIMARY KEY,
            generation INTEGER NOT NULL CHECK (generation >= 0)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS embedded_chunks (
            vector_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
            text TEXT NOT NULL,
            page_numbers_json TEXT NOT NULL,
            start_page INTEGER NOT NULL CHECK (start_page >= 1),
            end_page INTEGER NOT NULL CHECK (end_page >= 1),
            embedding BLOB NOT NULL
                CHECK (typeof(embedding) = 'blob' AND length(embedding) = 1536),
            dimension INTEGER NOT NULL CHECK (dimension = 384),
            encoding TEXT NOT NULL CHECK (encoding = 'float32-le-v1'),
            model_name TEXT NOT NULL,
            UNIQUE (owner_id, document_id, chunk_id),
            UNIQUE (owner_id, document_id, chunk_index),
            FOREIGN KEY (document_id, owner_id)
                REFERENCES documents(id, owner_id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_embedded_chunks_owner
        ON embedded_chunks(owner_id, vector_id)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_embedded_chunks_document
        ON embedded_chunks(owner_id, document_id, vector_id)
        """
    )


def _migration_document_summaries(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS document_summary_cache (
            document_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            summary_version INTEGER NOT NULL CHECK (summary_version >= 1),
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (document_id, owner_id, content_sha256, summary_version),
            FOREIGN KEY (document_id, owner_id)
                REFERENCES documents(id, owner_id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS document_summary_jobs (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            content_sha256 TEXT,
            summary_version INTEGER NOT NULL CHECK (summary_version >= 1),
            filename TEXT NOT NULL,
            page_count INTEGER NOT NULL CHECK (page_count >= 0),
            text_status TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('queued', 'processing', 'completed', 'failed')
            ),
            phase TEXT NOT NULL,
            progress_current INTEGER NOT NULL DEFAULT 0
                CHECK (progress_current >= 0),
            progress_total INTEGER NOT NULL DEFAULT 0
                CHECK (progress_total >= 0),
            result_json TEXT,
            error_code TEXT,
            retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
            cache_hit INTEGER NOT NULL DEFAULT 0 CHECK (cache_hit IN (0, 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (document_id, owner_id)
                REFERENCES documents(id, owner_id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_document_summary_active_job
        ON document_summary_jobs(owner_id, document_id)
        WHERE status IN ('queued', 'processing')
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_document_summary_jobs_owner_document
        ON document_summary_jobs(owner_id, document_id, created_at DESC)
        """
    )


def _encode_embedding(vector: object) -> bytes:
    if not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding must contain exactly {EMBEDDING_DIMENSION} values."
        )
    values: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Embedding values must be finite numbers.")
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError("Embedding values must be finite numbers.")
        values.append(numeric_value)
    if not math.isclose(
        math.hypot(*values),
        1.0,
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError("Embedding must be L2-normalized.")
    try:
        return struct.pack(f"<{EMBEDDING_DIMENSION}f", *values)
    except (OverflowError, struct.error) as error:
        raise ValueError("Embedding values exceed float32 storage range.") from error


def _decode_embedding(blob: object) -> tuple[float, ...]:
    if not isinstance(blob, bytes) or len(blob) != _EMBEDDING_BLOB_SIZE:
        raise ValueError("Stored embedding has an invalid float32 BLOB size.")
    return struct.unpack(f"<{EMBEDDING_DIMENSION}f", blob)


def _validate_embedded_chunk(
    chunk: EmbeddedChunk, document_id: str, position: int
) -> tuple[object, ...]:
    if not isinstance(chunk, dict):
        raise ValueError(f"Embedded chunk at position {position} must be a mapping.")
    for field in ("chunk_id", "document_id", "text"):
        value = chunk.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Embedded chunk at position {position} must have non-empty {field}."
            )
    if chunk["document_id"] != document_id:
        raise ValueError(
            f"Embedded chunk at position {position} belongs to another document."
        )
    chunk_index = chunk.get("chunk_index")
    if (
        isinstance(chunk_index, bool)
        or not isinstance(chunk_index, int)
        or chunk_index < 0
    ):
        raise ValueError(
            f"Embedded chunk at position {position} must have a non-negative chunk_index."
        )
    page_numbers = chunk.get("page_numbers")
    if (
        not isinstance(page_numbers, list)
        or not page_numbers
        or any(
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
            for page_number in page_numbers
        )
    ):
        raise ValueError(
            f"Embedded chunk at position {position} must have positive page_numbers."
        )
    start_page = chunk.get("start_page")
    end_page = chunk.get("end_page")
    if (
        isinstance(start_page, bool)
        or not isinstance(start_page, int)
        or start_page != page_numbers[0]
        or isinstance(end_page, bool)
        or not isinstance(end_page, int)
        or end_page != page_numbers[-1]
    ):
        raise ValueError(
            f"Embedded chunk at position {position} has inconsistent page metadata."
        )
    return (
        chunk["chunk_id"],
        document_id,
        chunk_index,
        chunk["text"],
        json.dumps(page_numbers, separators=(",", ":")),
        start_page,
        end_page,
        _encode_embedding(chunk.get("embedding")),
    )


def _stored_embedding_record(row: sqlite3.Row) -> StoredEmbeddingRecord:
    page_numbers = json.loads(row["page_numbers_json"])
    if (
        not isinstance(page_numbers, list)
        or not page_numbers
        or any(
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
            for page_number in page_numbers
        )
        or page_numbers[0] != row["start_page"]
        or page_numbers[-1] != row["end_page"]
    ):
        raise ValueError("Stored embedding has invalid page metadata.")
    if row["dimension"] != EMBEDDING_DIMENSION:
        raise ValueError("Stored embedding has an unsupported dimension.")
    if row["encoding"] != EMBEDDING_ENCODING:
        raise ValueError("Stored embedding has an unsupported encoding.")
    embedding = _decode_embedding(row["embedding"])
    if any(not math.isfinite(value) for value in embedding):
        raise ValueError("Stored embedding contains non-finite values.")
    if not math.isclose(
        math.hypot(*embedding),
        1.0,
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError("Stored embedding is not L2-normalized.")
    return StoredEmbeddingRecord(
        vector_id=row["vector_id"],
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        document_name=row["document_name"],
        owner_id=row["owner_id"],
        chunk_index=row["chunk_index"],
        text=row["text"],
        page_numbers=tuple(page_numbers),
        start_page=row["start_page"],
        end_page=row["end_page"],
        embedding=embedding,
        dimension=row["dimension"],
        encoding=row["encoding"],
        model_name=row["model_name"],
    )


def _apply_migration(
    connection: sqlite3.Connection,
    version: int,
    action: Callable[[sqlite3.Connection], None],
) -> None:
    exists = connection.execute(
        "SELECT 1 FROM schema_migrations WHERE version = ?", (version,)
    ).fetchone()
    if exists is not None:
        return

    connection.execute("BEGIN IMMEDIATE")
    try:
        exists = connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = ?", (version,)
        ).fetchone()
        if exists is None:
            action(connection)
            connection.execute(
                "INSERT INTO schema_migrations (version) VALUES (?)", (version,)
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise


class SQLiteDocumentRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @_translate_sqlite_errors
    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            migration_table = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'schema_migrations'
                """
            ).fetchone()
            if migration_table is None:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS schema_migrations (
                            version INTEGER PRIMARY KEY,
                            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )
                        """
                    )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
            _apply_migration(connection, 1, _migration_baseline)
            _apply_migration(connection, 2, _migration_document_metadata)
            _apply_migration(connection, 3, _migration_document_pages)
            _apply_migration(connection, 4, _migration_document_list_index)
            _apply_migration(connection, 5, _migration_embedded_chunks)
            _apply_migration(connection, 6, _migration_document_summaries)

    @staticmethod
    def _record(row: sqlite3.Row | None) -> DocumentRecord | None:
        if row is None:
            return None
        return DocumentRecord(
            id=row["id"],
            owner_id=row["owner_id"],
            filename=row["filename"],
            storage_key=row["stored_filename"],
            uploaded_at=row["uploaded_at"],
            size_bytes=row["size_bytes"],
            status=row["status"],
            page_count=row["page_count"],
            text_status=row["text_status"],
            idempotency_key=row["idempotency_key"],
            content_sha256=row["content_sha256"],
            staging_key=row["staging_filename"],
        )

    @_translate_sqlite_errors
    def find_by_idempotency_key(
        self, owner_id: str, idempotency_key: str
    ) -> DocumentRecord | None:
        self.initialize()
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM documents
                WHERE owner_id = ? AND idempotency_key = ?
                """,
                (owner_id, idempotency_key),
            ).fetchone()
        return self._record(row)

    @_translate_sqlite_errors
    def create_pending(
        self, document: NewDocument, pages: list[dict[str, str | int]]
    ) -> None:
        self.initialize()
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO documents (
                        id, owner_id, filename, stored_filename, uploaded_at,
                        size_bytes, status, page_count, text_status,
                        idempotency_key, content_sha256, staging_filename
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.owner_id,
                        document.filename,
                        document.storage_key,
                        document.uploaded_at,
                        document.size_bytes,
                        document.page_count,
                        document.text_status,
                        document.idempotency_key,
                        document.content_sha256,
                        document.staging_key,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO document_pages (document_id, page_number, text)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (document.id, page["page_number"], page["text"])
                        for page in pages
                    ],
                )
        except sqlite3.IntegrityError as error:
            existing = self.find_by_idempotency_key(
                document.owner_id, document.idempotency_key
            )
            if existing is not None:
                raise DuplicateIdempotencyKeyError from error
            raise

    @_translate_sqlite_errors
    def mark_ready(self, owner_id: str, document_id: str) -> None:
        self.initialize()
        with closing(self._connect()) as connection, connection:
            result = connection.execute(
                """
                UPDATE documents
                SET status = 'ready', staging_filename = NULL
                WHERE id = ? AND owner_id = ? AND status = 'pending'
                """,
                (document_id, owner_id),
            )
            if result.rowcount != 1:
                raise sqlite3.DatabaseError("Pending document could not be finalized")

    @_translate_sqlite_errors
    def delete_pending(self, owner_id: str, document_id: str) -> None:
        self.initialize()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                """
                DELETE FROM documents
                WHERE id = ? AND owner_id = ? AND status = 'pending'
                """,
                (document_id, owner_id),
            )

    @_translate_sqlite_errors
    def list_pending(self) -> list[DocumentRecord]:
        self.initialize()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM documents WHERE status = 'pending'"
            ).fetchall()
        return [record for row in rows if (record := self._record(row)) is not None]

    @staticmethod
    def _decode_cursor(cursor: str) -> tuple[str, str]:
        try:
            value = json.loads(
                base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
            )
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not all(isinstance(part, str) for part in value)
            ):
                raise ValueError
            return value[0], value[1]
        except (ValueError, UnicodeError, binascii.Error) as error:
            raise InvalidDocumentCursorError("Invalid document cursor.") from error

    @staticmethod
    def _encode_cursor(uploaded_at: str, document_id: str) -> str:
        payload = json.dumps([uploaded_at, document_id], separators=(",", ":"))
        return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")

    @_translate_sqlite_errors
    def list_documents(
        self, owner_id: str, limit: int, cursor: str | None
    ) -> tuple[list[dict[str, str | int | None]], str | None, int]:
        self.initialize()
        parameters: list[object] = [owner_id]
        cursor_clause = ""
        if cursor:
            uploaded_at, document_id = self._decode_cursor(cursor)
            cursor_clause = "AND (uploaded_at < ? OR (uploaded_at = ? AND id < ?))"
            parameters.extend((uploaded_at, uploaded_at, document_id))
        parameters.append(limit + 1)

        with closing(self._connect()) as connection:
            connection.execute("BEGIN")
            total_count = connection.execute(
                """
                SELECT COUNT(*) FROM documents
                WHERE owner_id = ? AND status = 'ready'
                """,
                (owner_id,),
            ).fetchone()[0]
            rows = connection.execute(
                f"""
                SELECT id, filename, uploaded_at, size_bytes, page_count, text_status
                FROM documents
                WHERE owner_id = ? AND status = 'ready' {cursor_clause}
                ORDER BY uploaded_at DESC, id DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()

        has_more = len(rows) > limit
        page_rows = rows[:limit]
        result = [dict(row) for row in page_rows]
        next_cursor = None
        if has_more and page_rows:
            last_row = page_rows[-1]
            next_cursor = self._encode_cursor(last_row["uploaded_at"], last_row["id"])
        return result, next_cursor, total_count

    @_translate_sqlite_errors
    def get_document_pages(
        self, owner_id: str, document_id: str
    ) -> list[dict[str, str | int]] | None:
        self.initialize()
        with closing(self._connect()) as connection:
            document = connection.execute(
                """
                SELECT 1 FROM documents
                WHERE id = ? AND owner_id = ? AND status = 'ready'
                """,
                (document_id, owner_id),
            ).fetchone()
            if document is None:
                return None
            rows = connection.execute(
                """
                SELECT page_number, text FROM document_pages
                WHERE document_id = ? ORDER BY page_number
                """,
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @_translate_sqlite_errors
    def get_document_pages_for_indexing(
        self, owner_id: str, document_id: str
    ) -> list[dict[str, str | int]] | None:
        self.initialize()
        with closing(self._connect()) as connection:
            document = connection.execute(
                """
                SELECT 1 FROM documents
                WHERE id = ? AND owner_id = ? AND status = 'pending'
                """,
                (document_id, owner_id),
            ).fetchone()
            if document is None:
                return None
            rows = connection.execute(
                """
                SELECT page_number, text FROM document_pages
                WHERE document_id = ? ORDER BY page_number
                """,
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _require_ready_document(
        connection: sqlite3.Connection, owner_id: str, document_id: str
    ) -> None:
        row = connection.execute(
            """
            SELECT 1 FROM documents
            WHERE id = ? AND owner_id = ? AND status = 'ready'
            """,
            (document_id, owner_id),
        ).fetchone()
        if row is None:
            raise DocumentNotReadyError(
                "A ready document for this owner was not found."
            )

    @staticmethod
    def _require_indexable_document(
        connection: sqlite3.Connection, owner_id: str, document_id: str
    ) -> str:
        row = connection.execute(
            """
            SELECT status FROM documents
            WHERE id = ? AND owner_id = ? AND status IN ('pending', 'ready')
            """,
            (document_id, owner_id),
        ).fetchone()
        if row is None:
            raise DocumentNotReadyError(
                "An indexable document for this owner was not found."
            )
        return row["status"]

    @staticmethod
    def _increment_owner_generation(
        connection: sqlite3.Connection, owner_id: str
    ) -> int:
        connection.execute(
            """
            INSERT INTO owner_embedding_generations (owner_id, generation)
            VALUES (?, 1)
            ON CONFLICT(owner_id) DO UPDATE
            SET generation = generation + 1
            """,
            (owner_id,),
        )
        row = connection.execute(
            """
            SELECT generation FROM owner_embedding_generations
            WHERE owner_id = ?
            """,
            (owner_id,),
        ).fetchone()
        return int(row["generation"])

    @_translate_sqlite_errors
    def replace_document_embeddings(
        self,
        owner_id: str,
        document_id: str,
        model_name: str,
        chunks: list[EmbeddedChunk],
    ) -> int:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must not be empty.")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must not be empty.")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must not be empty.")

        rows = [
            _validate_embedded_chunk(chunk, document_id, position)
            for position, chunk in enumerate(chunks)
        ]
        self.initialize()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                status = self._require_indexable_document(
                    connection, owner_id, document_id
                )
                connection.execute(
                    """
                    DELETE FROM embedded_chunks
                    WHERE owner_id = ? AND document_id = ?
                    """,
                    (owner_id, document_id),
                )
                connection.executemany(
                    """
                    INSERT INTO embedded_chunks (
                        chunk_id, document_id, owner_id, chunk_index, text,
                        page_numbers_json, start_page, end_page, embedding,
                        dimension, encoding, model_name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            *row[:2],
                            owner_id,
                            *row[2:],
                            EMBEDDING_DIMENSION,
                            EMBEDDING_ENCODING,
                            model_name,
                        )
                        for row in rows
                    ],
                )
                if status == "pending":
                    connection.execute(
                        """
                        UPDATE documents
                        SET status = 'ready', staging_filename = NULL
                        WHERE id = ? AND owner_id = ? AND status = 'pending'
                        """,
                        (document_id, owner_id),
                    )
                generation = self._increment_owner_generation(connection, owner_id)
                connection.commit()
                return generation
            except Exception:
                connection.rollback()
                raise

    @_translate_sqlite_errors
    def get_owner_embeddings(self, owner_id: str) -> list[StoredEmbeddingRecord]:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must not be empty.")
        self.initialize()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT embedded_chunks.*, documents.filename AS document_name
                FROM embedded_chunks
                JOIN documents
                  ON documents.id = embedded_chunks.document_id
                 AND documents.owner_id = embedded_chunks.owner_id
                WHERE embedded_chunks.owner_id = ?
                  AND documents.owner_id = ?
                  AND documents.status = 'ready'
                ORDER BY embedded_chunks.vector_id
                """,
                (owner_id, owner_id),
            ).fetchall()
        return [_stored_embedding_record(row) for row in rows]

    @_translate_sqlite_errors
    def resolve_embedding_ids(
        self, owner_id: str, vector_ids: list[int]
    ) -> list[StoredEmbeddingRecord]:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must not be empty.")
        if any(
            isinstance(vector_id, bool)
            or not isinstance(vector_id, int)
            or vector_id < 1
            for vector_id in vector_ids
        ):
            raise ValueError("vector_ids must contain positive integers.")
        if not vector_ids:
            return []

        self.initialize()
        placeholders = ",".join("?" for _ in vector_ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT embedded_chunks.*, documents.filename AS document_name
                FROM embedded_chunks
                JOIN documents
                  ON documents.id = embedded_chunks.document_id
                 AND documents.owner_id = embedded_chunks.owner_id
                WHERE embedded_chunks.owner_id = ?
                  AND documents.owner_id = ?
                  AND documents.status = 'ready'
                  AND embedded_chunks.vector_id IN ({placeholders})
                """,
                [owner_id, owner_id, *vector_ids],
            ).fetchall()
        by_id = {row["vector_id"]: _stored_embedding_record(row) for row in rows}
        return [by_id[vector_id] for vector_id in vector_ids if vector_id in by_id]

    @_translate_sqlite_errors
    def remove_document_embeddings(self, owner_id: str, document_id: str) -> int:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must not be empty.")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must not be empty.")
        self.initialize()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_ready_document(connection, owner_id, document_id)
                result = connection.execute(
                    """
                    DELETE FROM embedded_chunks
                    WHERE owner_id = ? AND document_id = ?
                    """,
                    (owner_id, document_id),
                )
                if result.rowcount:
                    generation = self._increment_owner_generation(connection, owner_id)
                else:
                    row = connection.execute(
                        """
                        SELECT generation FROM owner_embedding_generations
                        WHERE owner_id = ?
                        """,
                        (owner_id,),
                    ).fetchone()
                    generation = int(row["generation"]) if row else 0
                connection.commit()
                return generation
            except Exception:
                connection.rollback()
                raise

    @_translate_sqlite_errors
    def get_owner_index_generation(self, owner_id: str) -> int:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must not be empty.")
        self.initialize()
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT generation FROM owner_embedding_generations
                WHERE owner_id = ?
                """,
                (owner_id,),
            ).fetchone()
        return int(row["generation"]) if row else 0
