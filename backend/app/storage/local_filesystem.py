import hashlib
import functools
import os
import re
from pathlib import Path
from typing import BinaryIO

from app.documents import CHUNK_SIZE
from app.storage.interfaces import StagedFile, StorageOperationError

_DOCUMENT_KEY_PATTERN = re.compile(r"doc-[0-9a-f-]{36}\.pdf\Z")
_STAGING_KEY_PATTERN = re.compile(r"\.staging-[0-9a-f-]{36}\.part\Z")


class StorageIntegrityError(OSError):
    """Raised when a generated storage object does not match expected content."""


def _translate_os_errors(operation):
    @functools.wraps(operation)
    def wrapped(*args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except OSError as error:
            raise StorageOperationError("Document storage operation failed.") from error

    return wrapped


class LocalFilesystemStorage:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str, pattern: re.Pattern[str]) -> Path:
        if not pattern.fullmatch(key):
            raise StorageOperationError("Invalid generated storage key.")
        return self.root / key

    @staticmethod
    def _hash_path(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while chunk := file.read(CHUNK_SIZE):
                digest.update(chunk)
        return digest.hexdigest()

    @_translate_os_errors
    def stage(self, file: BinaryIO, document_id: str) -> StagedFile:
        self.root.mkdir(parents=True, exist_ok=True)
        storage_key = f"doc-{document_id}.pdf"
        staging_key = f".staging-{document_id}.part"
        staging_path = self._path(staging_key, _STAGING_KEY_PATTERN)
        digest = hashlib.sha256()
        size_bytes = 0

        file.seek(0)
        try:
            with staging_path.open("xb") as staged_file:
                while chunk := file.read(CHUNK_SIZE):
                    staged_file.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
                staged_file.flush()
                os.fsync(staged_file.fileno())
        except OSError:
            staging_path.unlink(missing_ok=True)
            raise
        finally:
            file.seek(0)

        return StagedFile(
            storage_key=storage_key,
            staging_key=staging_key,
            size_bytes=size_bytes,
            content_sha256=digest.hexdigest(),
        )

    @_translate_os_errors
    def promote(self, staged: StagedFile) -> None:
        staging_path = self._path(staged.staging_key, _STAGING_KEY_PATTERN)
        final_path = self._path(staged.storage_key, _DOCUMENT_KEY_PATTERN)
        os.link(staging_path, final_path)
        staging_path.unlink()

    @_translate_os_errors
    def recover(self, staged: StagedFile) -> bool:
        staging_path = self._path(staged.staging_key, _STAGING_KEY_PATTERN)
        final_path = self._path(staged.storage_key, _DOCUMENT_KEY_PATTERN)
        if final_path.exists():
            if self._hash_path(final_path) != staged.content_sha256:
                raise StorageIntegrityError("Stored PDF content hash does not match.")
            if staging_path.exists():
                if self._hash_path(staging_path) != staged.content_sha256:
                    raise StorageIntegrityError("Staged PDF content hash does not match.")
                staging_path.unlink()
            return True
        if not staging_path.exists():
            return False
        if self._hash_path(staging_path) != staged.content_sha256:
            raise StorageIntegrityError("Staged PDF content hash does not match.")
        self.promote(staged)
        return True

    @_translate_os_errors
    def cleanup(self, staged: StagedFile) -> None:
        for key, pattern in (
            (staged.staging_key, _STAGING_KEY_PATTERN),
            (staged.storage_key, _DOCUMENT_KEY_PATTERN),
        ):
            path = self._path(key, pattern)
            if path.exists():
                if self._hash_path(path) != staged.content_sha256:
                    raise StorageIntegrityError(
                        "Refusing to remove a PDF with unexpected content."
                    )
                path.unlink()
