from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep

import pytest

from app.services.embeddings.interfaces import EmbeddingProvider
from app.services.embeddings.sentence_transformer import (
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    EXPECTED_EMBEDDING_DIMENSION,
    SentenceTransformerProvider,
)


def normalized_vector(first_value: float = 1.0) -> list[float]:
    return [first_value, *([0.0] * (EXPECTED_EMBEDDING_DIMENSION - 1))]


class MockSentenceTransformerModel:
    def __init__(self, response: object | None = None) -> None:
        self.response = response
        self.encode_calls: list[tuple[list[str], bool]] = []

    def encode(
        self, sentences: list[str], *, normalize_embeddings: bool
    ) -> object:
        self.encode_calls.append((list(sentences), normalize_embeddings))
        if self.response is not None:
            return self.response
        return [normalized_vector() for _ in sentences]


class RecordingModelFactory:
    def __init__(self, model: MockSentenceTransformerModel | None = None) -> None:
        self.model = model or MockSentenceTransformerModel()
        self.calls: list[tuple[str, str]] = []

    def __call__(self, model_name: str, device: str) -> MockSentenceTransformerModel:
        self.calls.append((model_name, device))
        return self.model


def test_model_is_lazy_until_first_nonempty_embedding_request() -> None:
    factory = RecordingModelFactory()
    provider = SentenceTransformerProvider(model_factory=factory)

    assert factory.calls == []
    assert provider.embed_texts([]) == []
    assert factory.calls == []

    vectors = provider.embed_texts(["first"])

    assert factory.calls == [(DEFAULT_MODEL_NAME, DEFAULT_DEVICE)]
    assert len(vectors) == 1


def test_model_is_reused_and_encode_requests_normalized_embeddings() -> None:
    factory = RecordingModelFactory()
    provider: EmbeddingProvider = SentenceTransformerProvider(model_factory=factory)

    assert provider.model_identifier == provider.model_name
    provider.embed_texts(["first"])
    provider.embed_texts(["second", "third"])

    assert factory.calls == [(DEFAULT_MODEL_NAME, DEFAULT_DEVICE)]
    assert factory.model.encode_calls == [
        (["first"], True),
        (["second", "third"], True),
    ]


def test_model_name_and_device_can_be_injected() -> None:
    factory = RecordingModelFactory()
    provider = SentenceTransformerProvider(
        model_name="test/model",
        device="cpu",
        model_factory=factory,
    )

    provider.embed_texts(["text"])

    assert factory.calls == [("test/model", "cpu")]


def test_output_values_are_converted_to_python_floats() -> None:
    response = [[1 if index == 0 else 0 for index in range(EXPECTED_EMBEDDING_DIMENSION)]]
    model = MockSentenceTransformerModel(response)
    provider = SentenceTransformerProvider(
        model_factory=RecordingModelFactory(model)
    )

    vectors = provider.embed_texts(["text"])

    assert all(type(value) is float for value in vectors[0])


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ([normalized_vector()], "returned 1 vectors; expected 2"),
        ([[1.0, 0.0], normalized_vector()], "expected 384"),
        (
            [[float("nan"), *([0.0] * 383)], normalized_vector()],
            "non-finite",
        ),
        (
            [[float("inf"), *([0.0] * 383)], normalized_vector()],
            "non-finite",
        ),
        (
            [[2.0, *([0.0] * 383)], normalized_vector()],
            "not L2-normalized",
        ),
    ],
)
def test_rejects_malformed_model_output(response: object, error: str) -> None:
    model = MockSentenceTransformerModel(response)
    provider = SentenceTransformerProvider(model_factory=RecordingModelFactory(model))

    with pytest.raises(ValueError, match=error):
        provider.embed_texts(["first", "second"])


def test_concurrent_first_requests_create_only_one_model() -> None:
    calls = 0
    calls_lock = Lock()
    model = MockSentenceTransformerModel()

    def factory(model_name: str, device: str) -> MockSentenceTransformerModel:
        nonlocal calls
        assert model_name == DEFAULT_MODEL_NAME
        assert device == DEFAULT_DEVICE
        with calls_lock:
            calls += 1
        sleep(0.02)
        return model

    provider = SentenceTransformerProvider(model_factory=factory)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(provider.embed_texts, [["text"]] * 8))

    assert calls == 1
    assert len(results) == 8
    assert len(model.encode_calls) == 8


def test_rejects_non_string_input_before_loading_model() -> None:
    factory = RecordingModelFactory()
    provider = SentenceTransformerProvider(model_factory=factory)

    with pytest.raises(ValueError, match="Every input text must be a string"):
        provider.embed_texts(["valid", None])  # type: ignore[list-item]

    assert factory.calls == []
