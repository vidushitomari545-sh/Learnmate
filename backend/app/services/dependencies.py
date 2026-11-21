from collections.abc import Callable
from threading import Lock

from app import config
from app.services.embeddings.interfaces import EmbeddingProvider
from app.services.embeddings.sentence_transformer import SentenceTransformerProvider
from app.services.embeddings.service import EmbeddingService
from app.repositories.interfaces import DocumentRepository
from app.repositories.sqlite_documents import SQLiteDocumentRepository
from app.repositories.sqlite_summaries import SQLiteSummaryRepository
from app.services.document_service import DocumentService
from app.services.document_summary import DocumentSummaryService
from app.services.document_summary_jobs import DocumentSummaryJobService
from app.services.llm.fake import FakeLLMProvider
from app.services.llm.gemini_provider import GeminiProvider
from app.services.llm.interfaces import LLMProvider
from app.services.llm.ollama_provider import OllamaProvider
from app.services.llm.openai_provider import OpenAIProvider
from app.services.rag_service import RAGService
from app.services.semantic_search import SemanticSearchService
from app.services.vector_index import VectorIndexService
from app.storage.interfaces import DocumentStorage
from app.storage.local_filesystem import LocalFilesystemStorage


class EmbeddingServiceFactory:
    def __init__(
        self,
        provider_factory: Callable[..., EmbeddingProvider] | None = None,
    ) -> None:
        self._provider_factory = provider_factory or SentenceTransformerProvider
        self._service: EmbeddingService | None = None
        self._lock = Lock()

    def get(self) -> EmbeddingService:
        if self._service is None:
            with self._lock:
                if self._service is None:
                    provider = self._provider_factory(
                        model_name=config.EMBEDDING_MODEL_NAME,
                        device=config.EMBEDDING_DEVICE,
                        dimension=config.EMBEDDING_DIMENSION,
                    )
                    self._service = EmbeddingService(
                        provider,
                        batch_size=config.EMBEDDING_BATCH_SIZE,
                    )
        return self._service


_embedding_service_factory = EmbeddingServiceFactory()


def get_document_repository() -> DocumentRepository:
    return SQLiteDocumentRepository(config.DATABASE_PATH)


def get_document_storage() -> DocumentStorage:
    return LocalFilesystemStorage(config.DOCUMENT_STORAGE_ROOT)


def get_document_service() -> DocumentService:
    return DocumentService(
        get_document_repository(),
        get_document_storage(),
        get_embedding_service(),
    )


def get_development_owner_id() -> str:
    return config.DEVELOPMENT_OWNER_ID


def get_embedding_service() -> EmbeddingService:
    return _embedding_service_factory.get()


class SemanticSearchServiceFactory:
    def __init__(
        self,
        embedding_service_factory: Callable[[], EmbeddingService] | None = None,
        repository_factory: Callable[[], DocumentRepository] | None = None,
        vector_index_factory: Callable[
            [DocumentRepository], VectorIndexService
        ] | None = None,
    ) -> None:
        self._embedding_service_factory = embedding_service_factory
        self._repository_factory = repository_factory
        self._vector_index_factory = vector_index_factory or VectorIndexService
        self._service: SemanticSearchService | None = None
        self._lock = Lock()

    def get(self) -> SemanticSearchService:
        if self._service is None:
            with self._lock:
                if self._service is None:
                    embedding_service = (
                        self._embedding_service_factory()
                        if self._embedding_service_factory is not None
                        else get_embedding_service()
                    )
                    repository = (
                        self._repository_factory()
                        if self._repository_factory is not None
                        else get_document_repository()
                    )
                    vector_index_service = self._vector_index_factory(repository)
                    self._service = SemanticSearchService(
                        embedding_service,
                        vector_index_service,
                    )
        return self._service


_semantic_search_service_factory = SemanticSearchServiceFactory()


def get_semantic_search_service() -> SemanticSearchService:
    return _semantic_search_service_factory.get()


class RAGServiceFactory:
    def __init__(
        self,
        semantic_search_factory: Callable[[], SemanticSearchService] | None = None,
        provider_factory: Callable[[], LLMProvider] | None = None,
        openai_provider_factory: Callable[..., LLMProvider] | None = None,
        gemini_provider_factory: Callable[..., LLMProvider] | None = None,
        ollama_provider_factory: Callable[..., LLMProvider] | None = None,
    ) -> None:
        self._semantic_search_factory = (
            semantic_search_factory or get_semantic_search_service
        )
        self._provider_factory = provider_factory or self._configured_provider
        self._openai_provider_factory = openai_provider_factory or OpenAIProvider
        self._gemini_provider_factory = gemini_provider_factory or GeminiProvider
        self._ollama_provider_factory = ollama_provider_factory or OllamaProvider
        self._service: RAGService | None = None
        self._lock = Lock()

    def _configured_provider(self) -> LLMProvider:
        if config.LLM_PROVIDER == "fake":
            return FakeLLMProvider()
        if config.LLM_PROVIDER == "openai":
            return self._openai_provider_factory(
                model=config.LLM_MODEL,
                timeout_seconds=config.LLM_TIMEOUT_SECONDS,
                max_retries=config.LLM_MAX_RETRIES,
            )
        if config.LLM_PROVIDER == "gemini":
            return self._gemini_provider_factory(
                api_key=config.GEMINI_API_KEY,
                model=config.GEMINI_MODEL,
                timeout_seconds=config.LLM_TIMEOUT_SECONDS,
                max_retries=config.LLM_MAX_RETRIES,
            )
        if config.LLM_PROVIDER == "ollama":
            return self._ollama_provider_factory(
                base_url=config.OLLAMA_BASE_URL,
                model=config.OLLAMA_MODEL,
                timeout_seconds=config.LLM_TIMEOUT_SECONDS,
                max_retries=config.LLM_MAX_RETRIES,
            )
        raise ValueError("Unsupported language model provider configuration.")

    def get(self) -> RAGService:
        if self._service is None:
            with self._lock:
                if self._service is None:
                    self._service = RAGService(
                        self._semantic_search_factory(),
                        self._provider_factory(),
                        top_k=config.RAG_TOP_K,
                        max_context_tokens=config.RAG_MAX_CONTEXT_TOKENS,
                        max_context_bytes=config.RAG_MAX_CONTEXT_BYTES,
                        max_output_tokens=config.RAG_MAX_OUTPUT_TOKENS,
                        max_question_characters=config.RAG_MAX_QUESTION_CHARACTERS,
                    )
        return self._service


_rag_service_factory = RAGServiceFactory()


def get_rag_service() -> RAGService:
    return _rag_service_factory.get()


_document_summary_jobs_service: DocumentSummaryJobService | None = None
_document_summary_jobs_lock = Lock()


def get_document_summary_job_service() -> DocumentSummaryJobService:
    global _document_summary_jobs_service
    if _document_summary_jobs_service is None:
        with _document_summary_jobs_lock:
            if _document_summary_jobs_service is None:
                _document_summary_jobs_service = DocumentSummaryJobService(
                    SQLiteSummaryRepository(config.DATABASE_PATH),
                    get_document_service(),
                    DocumentSummaryService(
                        get_rag_service().llm_provider,
                        max_context_tokens=config.RAG_MAX_CONTEXT_TOKENS,
                        max_context_bytes=config.RAG_MAX_CONTEXT_BYTES,
                        max_output_tokens=config.DOCUMENT_SUMMARY_MAX_OUTPUT_TOKENS,
                    ),
                )
    return _document_summary_jobs_service
