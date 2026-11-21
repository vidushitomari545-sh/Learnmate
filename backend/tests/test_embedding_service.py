from copy import deepcopy
from math import hypot

import pytest

from app.services.embeddings.service import EmbeddingService
from app.services.text_chunker import TextChunk


class FakeEmbeddingProvider:
    model_identifier = "test-provider/model-v1"
    dimension = 2

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        vectors = []
        for text in texts:
            vector = [float(len(text)), 1.0]
            norm = hypot(*vector)
            vectors.append([value / norm for value in vector])
        return vectors


def make_chunk(chunk_index: int, text: str | None = None) -> TextChunk:
    return {
        "chunk_id": f"chunk-{chunk_index}",
        "document_id": "document-1",
        "chunk_index": chunk_index,
        "text": text or f"Text {chunk_index}",
        "page_numbers": [chunk_index + 1, chunk_index + 2],
        "start_page": chunk_index + 1,
        "end_page": chunk_index + 2,
    }


def test_embeds_single_chunk_and_preserves_metadata() -> None:
    provider = FakeEmbeddingProvider()
    service = EmbeddingService(provider)
    chunk = make_chunk(0)

    result = service.embed_chunks([chunk])

    assert service.model_identifier == provider.model_identifier
    assert provider.batches == [[chunk["text"]]]
    assert len(result) == 1
    assert {key: value for key, value in result[0].items() if key != "embedding"} == chunk
    assert len(result[0]["embedding"]) == provider.dimension
    assert hypot(*result[0]["embedding"]) == pytest.approx(1.0)


def test_embeds_multiple_chunks_in_input_order() -> None:
    provider = FakeEmbeddingProvider()
    chunks = [make_chunk(0, "a"), make_chunk(1, "longer text")]

    result = EmbeddingService(provider).embed_chunks(chunks)

    assert [item["chunk_id"] for item in result] == ["chunk-0", "chunk-1"]
    assert [vector[0] for vector in (item["embedding"] for item in result)] == [
        pytest.approx(1 / hypot(1, 1)),
        pytest.approx(11 / hypot(11, 1)),
    ]


def test_batches_using_configured_batch_size_and_keeps_order() -> None:
    provider = FakeEmbeddingProvider()
    chunks = [make_chunk(index) for index in range(5)]

    result = EmbeddingService(provider, batch_size=2).embed_chunks(chunks)

    assert provider.batches == [
        ["Text 0", "Text 1"],
        ["Text 2", "Text 3"],
        ["Text 4"],
    ]
    assert [item["chunk_index"] for item in result] == list(range(5))


def test_empty_input_does_not_call_provider() -> None:
    provider = FakeEmbeddingProvider()

    assert EmbeddingService(provider).embed_chunks([]) == []
    assert provider.batches == []


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_rejects_invalid_batch_size(batch_size: object) -> None:
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        EmbeddingService(FakeEmbeddingProvider(), batch_size=batch_size)  # type: ignore[arg-type]


def test_rejects_chunk_with_invalid_required_metadata() -> None:
    chunk = make_chunk(0)
    del chunk["page_numbers"]

    with pytest.raises(ValueError, match="page_numbers"):
        EmbeddingService(FakeEmbeddingProvider()).embed_chunks([chunk])


class FixedResponseProvider:
    model_identifier = "test-provider/model-v1"
    dimension = 2

    def __init__(self, response: object) -> None:
        self.response = response
        self.received_texts: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.received_texts.append(list(texts))
        return self.response  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ([], "expected 1"),
        ([[1.0, 0.0], [1.0, 0.0]], "expected 1"),
        ([[1.0]], "exactly 2 values"),
        ([[float("nan"), 0.0]], "non-finite or non-numeric"),
        ([[float("inf"), 0.0]], "non-finite or non-numeric"),
        ([[2.0, 0.0]], "not L2-normalized"),
    ],
)
def test_rejects_provider_contract_violations(
    response: object, error: str
) -> None:
    service = EmbeddingService(FixedResponseProvider(response))

    with pytest.raises(ValueError, match=error):
        service.embed_chunks([make_chunk(0)])


def test_does_not_mutate_original_chunks_or_nested_metadata() -> None:
    provider = FakeEmbeddingProvider()
    chunk = make_chunk(0)
    original = deepcopy(chunk)

    result = EmbeddingService(provider).embed_chunks([chunk])
    result[0]["page_numbers"].append(99)

    assert chunk == original
    assert "embedding" not in chunk
