from dataclasses import dataclass
from typing import BinaryIO, Protocol


@dataclass(frozen=True)
class StagedFile:
    storage_key: str
    staging_key: str
    size_bytes: int
    content_sha256: str


class StorageOperationError(RuntimeError):
    """Raised when a storage operation fails."""


class DocumentStorage(Protocol):
    def stage(self, file: BinaryIO, document_id: str) -> StagedFile: ...

    def promote(self, staged: StagedFile) -> None: ...

    def recover(self, staged: StagedFile) -> bool: ...

    def cleanup(self, staged: StagedFile) -> None: ...
