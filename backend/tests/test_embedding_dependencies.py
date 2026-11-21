from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from app import config
from app.services import dependencies
from app.services.dependencies import EmbeddingServiceFactory
from app.services.embeddings.sentence_transformer import SentenceTransformerProvider
from app.services.embeddings.service import EmbeddingService


class FakeEmbeddingProvider:
    def __init__(
        self, model_name: str, device: str, dimension: int
    ) -> None:
        self.model_name = model_name
        self.model_identifier = model_name
        self.device = device
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return []


def test_factory_uses_custom_settings_and_caches_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "EMBEDDING_MODEL_NAME", "custom/model")
    monkeypatch.setattr(config, "EMBEDDING_DEVICE", "cpu")
    monkeypatch.setattr(config, "EMBEDDING_DIMENSION", 512)
    monkeypatch.setattr(config, "EMBEDDING_BATCH_SIZE", 12)
    constructed_providers: list[FakeEmbeddingProvider] = []

    def provider_factory(model_name: str, device: str, dimension: int):
        provider = FakeEmbeddingProvider(model_name, device, dimension)
        constructed_providers.append(provider)
        return provider

    factory = EmbeddingServiceFactory(provider_factory=provider_factory)
    service = factory.get()

    assert isinstance(service, EmbeddingService)
    assert service is factory.get()
    assert service.batch_size == 12
    assert len(constructed_providers) == 1
    assert constructed_providers[0].model_name == "custom/model"
    assert constructed_providers[0].device == "cpu"
    assert constructed_providers[0].dimension == 512


def test_default_factory_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dependencies,
        "_embedding_service_factory",
        EmbeddingServiceFactory(),
    )

    service = dependencies.get_embedding_service()

    assert isinstance(service.provider, SentenceTransformerProvider)
    assert service.provider.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert service.provider.device == "cpu"
    assert service.provider.dimension == 384
    assert service.batch_size == 32
    assert service.provider._model is None


def test_module_getter_returns_the_same_service_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = EmbeddingServiceFactory(provider_factory=FakeEmbeddingProvider)
    monkeypatch.setattr(dependencies, "_embedding_service_factory", factory)

    first = dependencies.get_embedding_service()
    second = dependencies.get_embedding_service()

    assert first is second


def test_concurrent_factory_initialization_creates_one_service() -> None:
    provider_count = 0
    count_lock = Lock()

    def provider_factory(model_name: str, device: str, dimension: int):
        nonlocal provider_count
        with count_lock:
            provider_count += 1
        return FakeEmbeddingProvider(model_name, device, dimension)

    factory = EmbeddingServiceFactory(provider_factory=provider_factory)
    with ThreadPoolExecutor(max_workers=8) as executor:
        services = list(executor.map(lambda _: factory.get(), range(16)))

    assert provider_count == 1
    assert all(service is services[0] for service in services)
