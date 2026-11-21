from pathlib import Path

import pytest

from app.services import dependencies
from app.repositories.interfaces import NewDocument
from app.repositories.sqlite_documents import SQLiteDocumentRepository
from app.services.dependencies import SemanticSearchServiceFactory
from app.services.embeddings.service import EmbeddingService
from app.services.semantic_search import SemanticSearchService
from app.services.vector_index import (
    VECTOR_DIMENSION,
    VectorIndexService,
    VectorSearchResult,
)

MODEL_IDENTIFIER = "test-provider/model-v1"


class FakeEmbeddingProvider:
    dimension = VECTOR_DIMENSION
    model_identifier = MODEL_IDENTIFIER

    def __init__(self) -> None:
        self.received_texts: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.received_texts.append(list(texts))
        return [[1.0, *([0.0] * (self.dimension - 1))] for _ in texts]


class FakeVectorIndexService:
    def __init__(self, results: list[VectorSearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[str, list[float], str, int]] = []
        self.error: Exception | None = None

    def search(
        self,
        owner_id: str,
        query_embedding: list[float],
        model_identifier: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        self.calls.append((owner_id, query_embedding, model_identifier, top_k))
        if self.error is not None:
            raise self.error
        return self.results


def make_result(
    chunk_id: str,
    document_id: str,
    document_name: str,
    score: float,
) -> VectorSearchResult:
    return VectorSearchResult(
        vector_id=1,
        chunk_id=chunk_id,
        document_id=document_id,
        document_name=document_name,
        chunk_index=0,
        text=f"Text for {chunk_id}",
        page_numbers=(1, 2),
        start_page=1,
        end_page=2,
        score=score,
    )


def make_semantic_service(
    index: FakeVectorIndexService | None = None,
) -> tuple[
    SemanticSearchService,
    FakeEmbeddingProvider,
    FakeVectorIndexService,
]:
    provider = FakeEmbeddingProvider()
    vector_index = index or FakeVectorIndexService()
    return SemanticSearchService(EmbeddingService(provider), vector_index), provider, vector_index


def add_document(
    repository: SQLiteDocumentRepository,
    document_id: str,
    owner_id: str,
    filename: str,
) -> None:
    repository.create_pending(
        NewDocument(
            id=document_id,
            owner_id=owner_id,
            filename=filename,
            storage_key=f"stored-{document_id}.pdf",
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


def add_chunk(
    repository: SQLiteDocumentRepository,
    document_id: str,
    owner_id: str,
    chunk_id: str,
    vector: list[float],
) -> None:
    repository.replace_document_embeddings(
        owner_id,
        document_id,
        MODEL_IDENTIFIER,
        [
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": 0,
                "text": f"Text for {chunk_id}",
                "page_numbers": [1, 2],
                "start_page": 1,
                "end_page": 2,
                "embedding": vector,
            }
        ],
    )


def test_search_embeds_query_and_passes_model_identifier_unchanged() -> None:
    expected_results = [
        make_result("first", "doc-a", "a.pdf", 0.9),
        make_result("second", "doc-b", "b.pdf", 0.7),
    ]
    index = FakeVectorIndexService(expected_results)
    service, provider, _ = make_semantic_service(index)

    results = service.search("trusted-owner", "  explain vectors  ", top_k=2)

    assert provider.received_texts == [["explain vectors"]]
    assert index.calls == [
        (
            "trusted-owner",
            [1.0, *([0.0] * (VECTOR_DIMENSION - 1))],
            MODEL_IDENTIFIER,
            2,
        )
    ]
    assert results == expected_results
    assert [result.score for result in results] == [0.9, 0.7]
    assert results[0].document_name == "a.pdf"
    assert results[0].text == "Text for first"
    assert results[0].page_numbers == (1, 2)


@pytest.mark.parametrize("query", ["", " ", "\t\n"])
def test_rejects_empty_query_before_embedding(query: str) -> None:
    service, provider, index = make_semantic_service()

    with pytest.raises(ValueError, match="query must not be empty"):
        service.search("owner-a", query)

    assert provider.received_texts == []
    assert index.calls == []


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_rejects_invalid_top_k_before_embedding(top_k: object) -> None:
    service, provider, index = make_semantic_service()

    with pytest.raises(ValueError, match="top_k"):
        service.search("owner-a", "query", top_k)  # type: ignore[arg-type]

    assert provider.received_texts == []
    assert index.calls == []


def test_empty_index_returns_empty_results(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider()
    repository = SQLiteDocumentRepository(tmp_path / "empty-semantic.sqlite3")
    service = SemanticSearchService(
        EmbeddingService(provider),
        VectorIndexService(repository),
    )

    assert service.search("owner-a", "query") == []
    assert provider.received_texts == [["query"]]


def test_propagates_model_mismatch_from_vector_index() -> None:
    index = FakeVectorIndexService()
    index.error = ValueError("Query model identifier does not match the indexed model.")
    service, _, _ = make_semantic_service(index)

    with pytest.raises(ValueError, match="does not match the indexed model"):
        service.search("owner-a", "query")


def test_sqlite_search_returns_only_ready_owner_results_with_names(
    tmp_path: Path,
) -> None:
    repository = SQLiteDocumentRepository(tmp_path / "semantic.sqlite3")
    add_document(repository, "a-best", "owner-a", "best.pdf")
    add_document(repository, "a-next", "owner-a", "next.pdf")
    add_document(repository, "a-pending", "owner-a", "pending.pdf")
    add_document(repository, "b-private", "owner-b", "private.pdf")

    exact_vector = [1.0, *([0.0] * (VECTOR_DIMENSION - 1))]
    similar_vector = [0.8, 0.6, *([0.0] * (VECTOR_DIMENSION - 2))]
    add_chunk(repository, "a-best", "owner-a", "best-chunk", exact_vector)
    add_chunk(repository, "a-next", "owner-a", "next-chunk", similar_vector)
    add_chunk(repository, "a-pending", "owner-a", "pending-chunk", exact_vector)
    add_chunk(repository, "b-private", "owner-b", "private-chunk", exact_vector)

    with repository._connect() as connection:
        connection.execute(
            "UPDATE documents SET status = 'pending' WHERE id = 'a-pending'"
        )

    provider = FakeEmbeddingProvider()
    service = SemanticSearchService(
        EmbeddingService(provider),
        VectorIndexService(repository),
    )

    results = service.search("owner-a", "find the relevant text", top_k=3)

    assert [result.chunk_id for result in results] == ["best-chunk", "next-chunk"]
    assert [result.document_name for result in results] == ["best.pdf", "next.pdf"]
    assert [result.document_id for result in results] == ["a-best", "a-next"]
    assert [result.score for result in results] == pytest.approx([1.0, 0.8])
    assert all(result.chunk_id not in {"pending-chunk", "private-chunk"} for result in results)


def test_sqlite_search_rejects_mismatched_embedding_model(
    tmp_path: Path,
) -> None:
    repository = SQLiteDocumentRepository(tmp_path / "mismatched-semantic.sqlite3")
    add_document(repository, "doc-one", "owner-a", "document.pdf")
    repository.replace_document_embeddings(
        "owner-a",
        "doc-one",
        "stored-model",
        [
            {
                "chunk_id": "chunk-one",
                "document_id": "doc-one",
                "chunk_index": 0,
                "text": "Stored text",
                "page_numbers": [1, 2],
                "start_page": 1,
                "end_page": 2,
                "embedding": [1.0, *([0.0] * (VECTOR_DIMENSION - 1))],
            }
        ],
    )
    service = SemanticSearchService(
        EmbeddingService(FakeEmbeddingProvider()),
        VectorIndexService(repository),
    )

    with pytest.raises(ValueError, match="does not match the indexed model"):
        service.search("owner-a", "query")


def test_semantic_search_factory_reuses_service_and_vector_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeEmbeddingProvider()
    embedding_service = EmbeddingService(provider)
    repository = object()
    index = FakeVectorIndexService()
    repository_count = 0
    index_count = 0

    def repository_factory() -> object:
        nonlocal repository_count
        repository_count += 1
        return repository

    def vector_index_factory(received_repository: object) -> FakeVectorIndexService:
        nonlocal index_count
        index_count += 1
        assert received_repository is repository
        return index

    factory = SemanticSearchServiceFactory(
        embedding_service_factory=lambda: embedding_service,
        repository_factory=repository_factory,  # type: ignore[arg-type]
        vector_index_factory=vector_index_factory,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(dependencies, "_semantic_search_service_factory", factory)

    first = dependencies.get_semantic_search_service()
    second = dependencies.get_semantic_search_service()

    assert first is second
    assert first.embedding_service is embedding_service
    assert first.vector_index_service is index
    assert repository_count == 1
    assert index_count == 1
