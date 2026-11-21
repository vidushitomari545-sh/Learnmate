"""Semantic retrieval over owner-scoped embedded document chunks."""

from app.services.embeddings.service import EmbeddingService
from app.services.vector_index import VectorIndexService, VectorSearchResult


class SemanticSearchService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_index_service: VectorIndexService,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_index_service = vector_index_service

    def search(
        self,
        owner_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must not be empty.")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty.")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer.")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.embedding_service.embed_query(query)
        return self.vector_index_service.search(
            owner_id,
            query_embedding,
            self.embedding_service.model_identifier,
            int(top_k),
        )
