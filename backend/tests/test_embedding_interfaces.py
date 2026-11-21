from math import isclose, sqrt

from app.services.embeddings import EmbeddedChunk, EmbeddingProvider


class FakeEmbeddingProvider:
    model_identifier = "test-provider/model-v1"
    dimension = 3

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            values = [float(len(text)), 1.0, 1.0]
            norm = sqrt(sum(value * value for value in values))
            vectors.append([value / norm for value in values])
        return vectors


def test_fake_provider_satisfies_embedding_provider_contract() -> None:
    provider: EmbeddingProvider = FakeEmbeddingProvider()

    assert provider.embed_texts([]) == []
    texts = ["first", "second"]
    vectors = provider.embed_texts(texts)

    assert len(vectors) == len(texts)
    assert all(len(vector) == provider.dimension for vector in vectors)
    assert vectors[0][0] < vectors[1][0]
    assert all(
        isclose(sqrt(sum(value * value for value in vector)), 1.0)
        for vector in vectors
    )


def test_embedded_chunk_preserves_text_chunk_metadata() -> None:
    chunk: EmbeddedChunk = {
        "chunk_id": "chunk-123",
        "document_id": "document-456",
        "chunk_index": 2,
        "text": "Source text",
        "page_numbers": [3, 4],
        "start_page": 3,
        "end_page": 4,
        "embedding": [1.0, 0.0, 0.0],
    }

    assert chunk == {
        "chunk_id": "chunk-123",
        "document_id": "document-456",
        "chunk_index": 2,
        "text": "Source text",
        "page_numbers": [3, 4],
        "start_page": 3,
        "end_page": 4,
        "embedding": [1.0, 0.0, 0.0],
    }
