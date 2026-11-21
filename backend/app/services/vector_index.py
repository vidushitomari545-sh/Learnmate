"""Owner-scoped, rebuildable FAISS indexes backed by SQLite vectors."""

from collections.abc import Callable
from dataclasses import dataclass
from math import hypot, isclose, isfinite
from numbers import Real
from threading import Lock
from typing import Protocol

import faiss
import numpy as np

from app.repositories.interfaces import DocumentRepository, StoredEmbeddingRecord

VECTOR_DIMENSION = 384
_NORMALIZATION_TOLERANCE = 1e-5
_GENERATION_READ_ATTEMPTS = 3
_SEARCH_GENERATION_RETRIES = 3


class _SearchIndex(Protocol):
    @property
    def ntotal(self) -> int: ...

    def add(self, vectors: np.ndarray) -> None: ...

    def search(
        self, queries: np.ndarray, count: int
    ) -> tuple[np.ndarray, np.ndarray]: ...


IndexFactory = Callable[[int], _SearchIndex]


@dataclass(frozen=True)
class VectorSearchResult:
    vector_id: int
    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    text: str
    page_numbers: tuple[int, ...]
    start_page: int
    end_page: int
    score: float


@dataclass(frozen=True)
class _OwnerIndex:
    generation: int
    index: _SearchIndex
    vector_ids: tuple[int, ...]
    model_identifier: str | None


class VectorIndexService:
    def __init__(
        self,
        repository: DocumentRepository,
        dimension: int = VECTOR_DIMENSION,
        index_factory: IndexFactory | None = None,
    ) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
            raise ValueError("dimension must be a positive integer.")
        if dimension != VECTOR_DIMENSION:
            raise ValueError(
                f"FAISS vector persistence requires dimension {VECTOR_DIMENSION}."
            )
        self.repository = repository
        self.dimension = dimension
        self._index_factory = index_factory or (
            lambda size: faiss.IndexFlatIP(size)
        )
        self._cache: dict[str, _OwnerIndex] = {}
        self._owner_locks: dict[str, Lock] = {}
        self._owner_locks_guard = Lock()

    def _owner_lock(self, owner_id: str) -> Lock:
        with self._owner_locks_guard:
            lock = self._owner_locks.get(owner_id)
            if lock is None:
                lock = Lock()
                self._owner_locks[owner_id] = lock
            return lock

    def _validate_owner_id(self, owner_id: str) -> None:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must not be empty.")

    def _validate_record(
        self, record: StoredEmbeddingRecord, position: int
    ) -> None:
        if record.dimension != self.dimension:
            raise ValueError(
                f"Stored vector {record.vector_id} has dimension {record.dimension}; "
                f"expected {self.dimension}."
            )
        if not isinstance(record.embedding, (tuple, list)):
            raise ValueError(f"Stored vector {record.vector_id} has a malformed vector.")
        if len(record.embedding) != self.dimension:
            raise ValueError(
                f"Stored vector {record.vector_id} has {len(record.embedding)} values; "
                f"expected {self.dimension}."
            )
        if (
            isinstance(record.vector_id, bool)
            or not isinstance(record.vector_id, int)
            or record.vector_id < 1
        ):
            raise ValueError(f"Stored vector at position {position} has an invalid ID.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            for value in record.embedding
        ):
            raise ValueError(
                f"Stored vector {record.vector_id} contains non-finite or "
                "non-numeric values."
            )
        if not isclose(
            hypot(*(float(value) for value in record.embedding)),
            1.0,
            rel_tol=_NORMALIZATION_TOLERANCE,
            abs_tol=_NORMALIZATION_TOLERANCE,
        ):
            raise ValueError(f"Stored vector {record.vector_id} is not normalized.")

    def _build_index(
        self, owner_id: str, generation: int, records: list[StoredEmbeddingRecord]
    ) -> _OwnerIndex:
        vector_ids: list[int] = []
        seen_ids: set[int] = set()
        model_names: set[str] = set()
        matrix: list[list[float]] = []
        for position, record in enumerate(records):
            if record.owner_id != owner_id:
                raise ValueError(
                    f"Repository returned vector {record.vector_id} for another owner."
                )
            self._validate_record(record, position)
            if record.vector_id in seen_ids:
                raise ValueError(f"Duplicate stored vector ID {record.vector_id}.")
            seen_ids.add(record.vector_id)
            vector_ids.append(record.vector_id)
            matrix.append([float(value) for value in record.embedding])
            if not isinstance(record.model_name, str) or not record.model_name.strip():
                raise ValueError(
                    f"Stored vector {record.vector_id} has no model identifier."
                )
            model_names.add(record.model_name)
        if len(model_names) > 1:
            raise ValueError("An owner's index cannot mix embedding models.")

        index = self._index_factory(self.dimension)
        if records:
            vectors = np.asarray(matrix, dtype=np.float32)
            if vectors.shape != (len(records), self.dimension):
                raise ValueError("Stored embeddings have an invalid matrix shape.")
            if not np.isfinite(vectors).all():
                raise ValueError("Stored embeddings exceed finite float32 values.")
            index.add(vectors)
            if index.ntotal != len(records):
                raise ValueError("FAISS index did not accept all stored embeddings.")
        model_identifier = next(iter(model_names)) if model_names else None
        return _OwnerIndex(generation, index, tuple(vector_ids), model_identifier)

    def _get_owner_index(self, owner_id: str) -> _OwnerIndex:
        for _ in range(_GENERATION_READ_ATTEMPTS):
            generation = self.repository.get_owner_index_generation(owner_id)
            cached = self._cache.get(owner_id)
            if cached is not None and cached.generation == generation:
                return cached

            records = self.repository.get_owner_embeddings(owner_id)
            confirmed_generation = self.repository.get_owner_index_generation(owner_id)
            if generation != confirmed_generation:
                continue
            rebuilt = self._build_index(owner_id, confirmed_generation, records)
            self._cache[owner_id] = rebuilt
            return rebuilt
        raise RuntimeError(
            "Owner embedding generation changed repeatedly while rebuilding the index."
        )

    @staticmethod
    def _validate_query(query_embedding: list[float], dimension: int) -> np.ndarray:
        if not isinstance(query_embedding, list) or len(query_embedding) != dimension:
            raise ValueError(f"Query embedding must contain exactly {dimension} values.")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not isfinite(value)
            for value in query_embedding
        ):
            raise ValueError("Query embedding must contain only finite numbers.")
        values = [float(value) for value in query_embedding]
        if not isclose(
            hypot(*values),
            1.0,
            rel_tol=_NORMALIZATION_TOLERANCE,
            abs_tol=_NORMALIZATION_TOLERANCE,
        ):
            raise ValueError("Query embedding must be L2-normalized.")
        query = np.asarray([values], dtype=np.float32)
        if not np.isfinite(query).all():
            raise ValueError("Query embedding exceeds finite float32 values.")
        return query

    def search(
        self,
        owner_id: str,
        query_embedding: list[float],
        model_identifier: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        self._validate_owner_id(owner_id)
        if not isinstance(model_identifier, str) or not model_identifier.strip():
            raise ValueError("model_identifier must not be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer.")
        if top_k < 0:
            raise ValueError("top_k must not be negative.")
        if top_k == 0:
            raise ValueError("top_k must be greater than zero.")
        query = self._validate_query(query_embedding, self.dimension)

        lock = self._owner_lock(owner_id)
        with lock:
            for _ in range(_SEARCH_GENERATION_RETRIES):
                owner_index = self._get_owner_index(owner_id)
                if (
                    owner_index.model_identifier is not None
                    and owner_index.model_identifier != model_identifier
                ):
                    raise ValueError(
                        "Query model identifier does not match the indexed model "
                        f"({owner_index.model_identifier!r})."
                    )
                if owner_index.vector_ids:
                    result_count = min(top_k, len(owner_index.vector_ids))
                    scores, positions = owner_index.index.search(query, result_count)
                    candidates: list[tuple[int, float]] = []
                    for position, score in zip(positions[0], scores[0]):
                        index_position = int(position)
                        numeric_score = float(score)
                        if index_position < 0:
                            continue
                        if index_position >= len(owner_index.vector_ids):
                            raise ValueError("FAISS returned an invalid result position.")
                        if not isfinite(numeric_score):
                            raise ValueError(
                                "FAISS returned a non-finite similarity score."
                            )
                        candidates.append(
                            (owner_index.vector_ids[index_position], numeric_score)
                        )

                    resolved = self.repository.resolve_embedding_ids(
                        owner_id,
                        [vector_id for vector_id, _ in candidates],
                    )
                    records_by_id = {record.vector_id: record for record in resolved}
                    results: list[VectorSearchResult] = []
                    for vector_id, score in candidates:
                        record = records_by_id.get(vector_id)
                        if record is None or record.owner_id != owner_id:
                            continue
                        results.append(
                            VectorSearchResult(
                                vector_id=record.vector_id,
                                chunk_id=record.chunk_id,
                                document_id=record.document_id,
                                document_name=record.document_name,
                                chunk_index=record.chunk_index,
                                text=record.text,
                                page_numbers=record.page_numbers,
                                start_page=record.start_page,
                                end_page=record.end_page,
                                score=score,
                            )
                        )
                else:
                    results = []

                if (
                    self.repository.get_owner_index_generation(owner_id)
                    == owner_index.generation
                ):
                    return results

            raise RuntimeError(
                "Owner embedding generation kept changing during search; "
                "retry the search."
            )
