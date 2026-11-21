from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Lock

import faiss
import pytest

from app.repositories.interfaces import NewDocument, StoredEmbeddingRecord
from app.repositories.sqlite_documents import (
    SQLiteDocumentRepository,
)
from app.services.embeddings.types import EmbeddedChunk
from app.services.vector_index import (
    VECTOR_DIMENSION,
    VectorIndexService,
    VectorSearchResult,
)

MODEL_IDENTIFIER = "sentence-transformers/all-MiniLM-L6-v2"


def vector_at(index: int, value: float = 1.0) -> tuple[float, ...]:
    values = [0.0] * VECTOR_DIMENSION
    values[index] = value
    return tuple(values)


def stored_record(
    vector_id: int,
    owner_id: str,
    chunk_id: str,
    document_id: str,
    embedding: tuple[float, ...],
    *,
    chunk_index: int = 0,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> StoredEmbeddingRecord:
    return StoredEmbeddingRecord(
        vector_id=vector_id,
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=f"{document_id}.pdf",
        owner_id=owner_id,
        chunk_index=chunk_index,
        text=f"source text {chunk_id}",
        page_numbers=(2, 4),
        start_page=2,
        end_page=4,
        embedding=embedding,
        dimension=VECTOR_DIMENSION,
        encoding="float32-le-v1",
        model_name=model_name,
    )


class FakeEmbeddingRepository:
    def __init__(self) -> None:
        self.records_by_owner: dict[str, list[StoredEmbeddingRecord]] = {}
        self.generations: dict[str, int] = {}
        self.generation_reads: dict[str, int] = {}
        self.record_reads: dict[str, int] = {}
        self.resolve_reads: list[tuple[str, list[int]]] = []
        self.after_resolve = None

    def get_owner_index_generation(self, owner_id: str) -> int:
        self.generation_reads[owner_id] = self.generation_reads.get(owner_id, 0) + 1
        return self.generations.get(owner_id, 0)

    def get_owner_embeddings(self, owner_id: str) -> list[StoredEmbeddingRecord]:
        self.record_reads[owner_id] = self.record_reads.get(owner_id, 0) + 1
        return list(self.records_by_owner.get(owner_id, []))

    def resolve_embedding_ids(
        self, owner_id: str, vector_ids: list[int]
    ) -> list[StoredEmbeddingRecord]:
        self.resolve_reads.append((owner_id, list(vector_ids)))
        records = {
            record.vector_id: record
            for record in self.records_by_owner.get(owner_id, [])
        }
        result = [records[vector_id] for vector_id in vector_ids if vector_id in records]
        if self.after_resolve is not None:
            hook = self.after_resolve
            self.after_resolve = None
            hook()
        return result


def make_service(
    repository: FakeEmbeddingRepository,
    index_factory=None,
) -> VectorIndexService:
    return VectorIndexService(repository, index_factory=index_factory)


def search_owner(
    service: VectorIndexService,
    owner_id: str,
    query_embedding: list[float],
    top_k: int = 5,
    model_identifier: str = MODEL_IDENTIFIER,
) -> list[VectorSearchResult]:
    return service.search(owner_id, query_embedding, model_identifier, top_k)


def create_ready_document(
    repository: SQLiteDocumentRepository, document_id: str, owner_id: str
) -> None:
    repository.create_pending(
        NewDocument(
            id=document_id,
            owner_id=owner_id,
            filename=f"{document_id}.pdf",
            storage_key=f"doc-{document_id}.pdf",
            staging_key=f".staging-{document_id}.part",
            uploaded_at="2026-11-01T00:00:00+00:00",
            size_bytes=1,
            page_count=2,
            text_status="extracted",
            idempotency_key=f"key-{document_id}",
            content_sha256=f"hash-{document_id}",
        ),
        [
            {"page_number": 1, "text": "first page"},
            {"page_number": 2, "text": "second page"},
        ],
    )
    repository.mark_ready(owner_id, document_id)


def make_embedded_chunk(
    document_id: str,
    chunk_id: str,
    chunk_index: int,
    vector: tuple[float, ...],
) -> EmbeddedChunk:
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "chunk_index": chunk_index,
        "text": f"Text for {chunk_id}",
        "page_numbers": [1, 2],
        "start_page": 1,
        "end_page": 2,
        "embedding": list(vector),
    }


def test_builds_index_and_returns_source_metadata_in_similarity_order() -> None:
    repository = FakeEmbeddingRepository()
    similar = [0.8, 0.6, *([0.0] * (VECTOR_DIMENSION - 2))]
    unrelated = vector_at(2)
    repository.records_by_owner["owner-a"] = [
        stored_record(20, "owner-a", "similar", "doc-a", tuple(similar)),
        stored_record(10, "owner-a", "unrelated", "doc-b", unrelated),
    ]
    service = make_service(repository)
    query = vector_at(0)

    results = search_owner(service, "owner-a", list(query), top_k=2)

    assert [result.vector_id for result in results] == [20, 10]
    assert [result.chunk_id for result in results] == ["similar", "unrelated"]
    assert [result.score for result in results] == pytest.approx([0.8, 0.0])
    assert results[0].document_id == "doc-a"
    assert results[0].page_numbers == (2, 4)
    assert results[0].start_page == 2
    assert results[0].end_page == 4
    assert repository.resolve_reads == [("owner-a", [20, 10])]


def test_mapping_uses_sorted_repository_records_not_vector_ids_as_positions() -> None:
    repository = FakeEmbeddingRepository()
    repository.records_by_owner["owner-a"] = [
        stored_record(91, "owner-a", "second", "doc", vector_at(1), chunk_index=1),
        stored_record(17, "owner-a", "first", "doc", vector_at(0), chunk_index=0),
    ]
    service = make_service(repository)

    results = search_owner(service, "owner-a", list(vector_at(1)), top_k=2)

    assert results[0].vector_id == 91
    assert results[0].chunk_id == "second"
    assert results[1].vector_id == 17
    assert results[1].chunk_id == "first"


def test_empty_index_returns_no_results_and_is_cached() -> None:
    repository = FakeEmbeddingRepository()
    service = make_service(repository)

    assert search_owner(service, "owner-empty", list(vector_at(0))) == []
    assert (
        search_owner(
            service,
            "owner-empty",
            list(vector_at(0)),
            model_identifier="a-different-model",
        )
        == []
    )
    assert search_owner(service, "owner-empty", list(vector_at(0))) == []

    assert repository.record_reads["owner-empty"] == 1
    assert repository.resolve_reads == []


def test_generation_change_rebuilds_index_from_sqlite() -> None:
    repository = FakeEmbeddingRepository()
    repository.generations["owner-a"] = 1
    repository.records_by_owner["owner-a"] = [
        stored_record(1, "owner-a", "old", "doc", vector_at(0))
    ]
    service = make_service(repository)

    first = search_owner(service, "owner-a", list(vector_at(0)))
    repository.generations["owner-a"] = 2
    repository.records_by_owner["owner-a"] = [
        stored_record(2, "owner-a", "new", "doc", vector_at(0))
    ]
    second = search_owner(service, "owner-a", list(vector_at(0)))

    assert first[0].chunk_id == "old"
    assert second[0].chunk_id == "new"
    assert repository.record_reads["owner-a"] == 2


def test_generation_change_during_resolution_discards_stale_results() -> None:
    repository = FakeEmbeddingRepository()
    repository.generations["owner-a"] = 1
    repository.records_by_owner["owner-a"] = [
        stored_record(1, "owner-a", "stale", "doc", vector_at(0))
    ]
    service = make_service(repository)

    def replace_embeddings() -> None:
        repository.generations["owner-a"] = 2
        repository.records_by_owner["owner-a"] = [
            stored_record(2, "owner-a", "fresh", "doc", vector_at(0))
        ]

    repository.after_resolve = replace_embeddings
    results = search_owner(service, "owner-a", list(vector_at(0)))

    assert [result.chunk_id for result in results] == ["fresh"]
    assert repository.record_reads["owner-a"] == 2
    assert repository.resolve_reads == [("owner-a", [1]), ("owner-a", [2])]


def test_search_stops_after_bounded_generation_retries() -> None:
    class AlwaysChangingRepository(FakeEmbeddingRepository):
        def resolve_embedding_ids(
            self, owner_id: str, vector_ids: list[int]
        ) -> list[StoredEmbeddingRecord]:
            resolved = super().resolve_embedding_ids(owner_id, vector_ids)
            self.generations[owner_id] = self.generations.get(owner_id, 0) + 1
            return resolved

    repository = AlwaysChangingRepository()
    repository.records_by_owner["owner-a"] = [
        stored_record(1, "owner-a", "chunk", "doc", vector_at(0))
    ]

    with pytest.raises(RuntimeError, match="generation kept changing during search"):
        search_owner(make_service(repository), "owner-a", list(vector_at(0)))

    assert len(repository.resolve_reads) == 3


def test_owner_isolation_and_repository_resolution_filter_other_owner() -> None:
    repository = FakeEmbeddingRepository()
    repository.records_by_owner["owner-a"] = [
        stored_record(1, "owner-a", "private-a", "doc-a", vector_at(0))
    ]
    repository.records_by_owner["owner-b"] = [
        stored_record(2, "owner-b", "private-b", "doc-b", vector_at(0))
    ]
    service = make_service(repository)

    results = search_owner(service, "owner-a", list(vector_at(0)))

    assert [result.chunk_id for result in results] == ["private-a"]
    assert repository.resolve_reads == [("owner-a", [1])]


def test_does_not_return_candidate_removed_or_no_longer_ready_in_repository() -> None:
    repository = FakeEmbeddingRepository()
    repository.records_by_owner["owner-a"] = [
        stored_record(1, "owner-a", "now-unavailable", "doc", vector_at(0))
    ]
    service = make_service(repository)
    assert search_owner(service, "owner-a", list(vector_at(0)))

    repository.records_by_owner["owner-a"] = []

    assert search_owner(service, "owner-a", list(vector_at(0))) == []


def test_rejects_repository_record_from_another_owner() -> None:
    repository = FakeEmbeddingRepository()
    repository.records_by_owner["owner-a"] = [
        stored_record(1, "owner-b", "wrong-owner", "doc-b", vector_at(0))
    ]

    with pytest.raises(ValueError, match="another owner"):
        search_owner(make_service(repository), "owner-a", list(vector_at(0)))


@pytest.mark.parametrize(
    ("record", "error"),
    [
        (
            replace(
                stored_record(1, "owner-a", "chunk", "doc", vector_at(0)),
                dimension=128,
            ),
            "dimension 128",
        ),
        (
            replace(
                stored_record(1, "owner-a", "chunk", "doc", vector_at(0)),
                embedding=(float("nan"), *([0.0] * (VECTOR_DIMENSION - 1))),
            ),
            "non-finite",
        ),
        (
            replace(
                stored_record(1, "owner-a", "chunk", "doc", vector_at(0)),
                embedding=(1.0, 0.0),
            ),
            "2 values",
        ),
        (
            replace(
                stored_record(1, "owner-a", "chunk", "doc", vector_at(0)),
                embedding=(2.0, *([0.0] * (VECTOR_DIMENSION - 1))),
            ),
            "not normalized",
        ),
        (
            replace(
                stored_record(1, "owner-a", "chunk", "doc", vector_at(0)),
                embedding=None,  # type: ignore[arg-type]
            ),
            "malformed vector",
        ),
    ],
)
def test_rejects_invalid_stored_vectors(
    record: StoredEmbeddingRecord, error: str
) -> None:
    repository = FakeEmbeddingRepository()
    repository.records_by_owner["owner-a"] = [record]

    with pytest.raises(ValueError, match=error):
        search_owner(make_service(repository), "owner-a", list(vector_at(0)))


def test_rejects_mixed_models_for_one_index() -> None:
    repository = FakeEmbeddingRepository()
    repository.records_by_owner["owner-a"] = [
        stored_record(1, "owner-a", "one", "doc", vector_at(0)),
        stored_record(
            2,
            "owner-a",
            "two",
            "doc",
            vector_at(1),
            model_name="other-model",
        ),
    ]

    with pytest.raises(ValueError, match="cannot mix embedding models"):
        search_owner(make_service(repository), "owner-a", list(vector_at(0)))


def test_rejects_query_model_that_differs_from_indexed_model() -> None:
    repository = FakeEmbeddingRepository()
    repository.records_by_owner["owner-a"] = [
        stored_record(1, "owner-a", "chunk", "doc", vector_at(0))
    ]

    with pytest.raises(ValueError, match="does not match the indexed model"):
        search_owner(
            make_service(repository),
            "owner-a",
            list(vector_at(0)),
            model_identifier="different-model",
        )


def test_rejects_configured_dimension_other_than_sqlite_dimension() -> None:
    with pytest.raises(ValueError, match="requires dimension 384"):
        VectorIndexService(FakeEmbeddingRepository(), dimension=128)


@pytest.mark.parametrize(
    ("query", "top_k", "error"),
    [
        ([1.0, 0.0], 1, "exactly 384"),
        ([float("inf"), *([0.0] * (VECTOR_DIMENSION - 1))], 1, "finite numbers"),
        ([2.0, *([0.0] * (VECTOR_DIMENSION - 1))], 1, "L2-normalized"),
    ],
)
def test_rejects_invalid_queries(
    query: list[float], top_k: int, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        search_owner(make_service(FakeEmbeddingRepository()), "owner-a", query, top_k)


@pytest.mark.parametrize(
    ("top_k", "error"),
    [
        (-1, "top_k must not be negative"),
        (0, "top_k must be greater than zero"),
    ],
)
def test_rejects_nonpositive_top_k(top_k: int, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        search_owner(
            make_service(FakeEmbeddingRepository()),
            "owner-a",
            list(vector_at(0)),
            top_k,
        )


def test_top_k_is_capped_to_available_vectors() -> None:
    repository = FakeEmbeddingRepository()
    repository.records_by_owner["owner-a"] = [
        stored_record(1, "owner-a", "first", "doc", vector_at(0)),
        stored_record(2, "owner-a", "second", "doc", vector_at(1)),
    ]

    results = search_owner(
        make_service(repository), "owner-a", list(vector_at(0)), top_k=50
    )

    assert len(results) == 2


def test_concurrent_initialization_builds_one_cached_index() -> None:
    repository = FakeEmbeddingRepository()
    repository.records_by_owner["owner-a"] = [
        stored_record(1, "owner-a", "chunk", "doc", vector_at(0))
    ]
    build_count = 0
    build_lock = Lock()

    def index_factory(dimension: int):
        nonlocal build_count
        with build_lock:
            build_count += 1
        return faiss.IndexFlatIP(dimension)

    service = make_service(repository, index_factory=index_factory)
    with ThreadPoolExecutor(max_workers=8) as executor:
        result_sets = list(
            executor.map(
                lambda _: search_owner(service, "owner-a", list(vector_at(0))),
                range(16),
            )
        )

    assert build_count == 1
    assert all(result_set[0].chunk_id == "chunk" for result_set in result_sets)


def test_concurrent_searches_rebuild_once_after_generation_change() -> None:
    repository = FakeEmbeddingRepository()
    repository.generations["owner-a"] = 1
    repository.records_by_owner["owner-a"] = [
        stored_record(1, "owner-a", "old", "doc", vector_at(0))
    ]
    service = make_service(repository)
    assert search_owner(service, "owner-a", list(vector_at(0)))[0].chunk_id == "old"

    repository.generations["owner-a"] = 2
    repository.records_by_owner["owner-a"] = [
        stored_record(2, "owner-a", "new", "doc", vector_at(0))
    ]
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: search_owner(service, "owner-a", list(vector_at(0))),
                range(16),
            )
        )

    assert repository.record_reads["owner-a"] == 2
    assert all(result_set[0].chunk_id == "new" for result_set in results)


def test_concurrent_searches_keep_different_owner_indexes_isolated() -> None:
    repository = FakeEmbeddingRepository()
    repository.records_by_owner["owner-a"] = [
        stored_record(1, "owner-a", "only-a", "doc-a", vector_at(0))
    ]
    repository.records_by_owner["owner-b"] = [
        stored_record(2, "owner-b", "only-b", "doc-b", vector_at(1))
    ]
    service = make_service(repository)
    requests = [("owner-a", vector_at(0)), ("owner-b", vector_at(1))] * 8

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda request: search_owner(service, request[0], list(request[1])),
                requests,
            )
        )

    for request, result_set in zip(requests, results):
        assert result_set[0].chunk_id == (
            "only-a" if request[0] == "owner-a" else "only-b"
        )


def test_sqlite_integration_enforces_owner_readiness_mapping_and_order(
    tmp_path: Path,
) -> None:
    repository = SQLiteDocumentRepository(tmp_path / "vectors.sqlite3")
    create_ready_document(repository, "doc-a-pending-later", "owner-a")
    create_ready_document(repository, "doc-a-ready", "owner-a")
    create_ready_document(repository, "doc-b-private", "owner-b")

    similar = (0.8, 0.6, *([0.0] * (VECTOR_DIMENSION - 2)))
    repository.replace_document_embeddings(
        "owner-a",
        "doc-a-pending-later",
        "model-v1",
        [make_embedded_chunk("doc-a-pending-later", "top-hit", 0, vector_at(0))],
    )
    repository.replace_document_embeddings(
        "owner-a",
        "doc-a-ready",
        "model-v1",
        [make_embedded_chunk("doc-a-ready", "second-hit", 0, similar)],
    )
    repository.replace_document_embeddings(
        "owner-b",
        "doc-b-private",
        "model-v1",
        [make_embedded_chunk("doc-b-private", "private-hit", 0, vector_at(0))],
    )

    service = VectorIndexService(repository)
    initial = search_owner(
        service, "owner-a", list(vector_at(0)), top_k=2, model_identifier="model-v1"
    )
    assert [result.chunk_id for result in initial] == ["top-hit", "second-hit"]

    with repository._connect() as connection:
        connection.execute(
            """
            UPDATE documents SET status = 'pending'
            WHERE id = 'doc-a-pending-later' AND owner_id = 'owner-a'
            """
        )

    owner_a_results = search_owner(
        service, "owner-a", list(vector_at(0)), top_k=2, model_identifier="model-v1"
    )
    owner_b_results = search_owner(
        service, "owner-b", list(vector_at(0)), top_k=2, model_identifier="model-v1"
    )

    assert [result.chunk_id for result in owner_a_results] == ["second-hit"]
    assert owner_a_results[0].document_id == "doc-a-ready"
    assert owner_a_results[0].page_numbers == (1, 2)
    assert [result.chunk_id for result in owner_b_results] == ["private-hit"]
