"""Provider-independent batching and validation for chunk embeddings."""

from math import hypot, isclose, isfinite
from numbers import Real

from app.services.embeddings.interfaces import EmbeddingProvider
from app.services.embeddings.types import EmbeddedChunk
from app.services.text_chunker import TextChunk

DEFAULT_EMBEDDING_BATCH_SIZE = 32
_NORMALIZATION_TOLERANCE = 1e-5


class EmbeddingService:
    def __init__(
        self,
        provider: EmbeddingProvider,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")
        model_identifier = getattr(provider, "model_identifier", None)
        if not isinstance(model_identifier, str) or not model_identifier.strip():
            raise ValueError("Embedding provider must declare a model_identifier.")
        self.provider = provider
        self.batch_size = batch_size
        self.model_identifier = model_identifier

    @staticmethod
    def _validate_chunk(chunk: TextChunk, position: int) -> None:
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk at position {position} must be a mapping.")

        for field in ("chunk_id", "document_id"):
            value = chunk.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Chunk at position {position} must have a non-empty {field}."
                )

        chunk_index = chunk.get("chunk_index")
        if (
            isinstance(chunk_index, bool)
            or not isinstance(chunk_index, int)
            or chunk_index < 0
        ):
            raise ValueError(
                f"Chunk at position {position} must have a non-negative integer chunk_index."
            )

        text = chunk.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Chunk at position {position} must have non-empty text.")

        page_numbers = chunk.get("page_numbers")
        if (
            not isinstance(page_numbers, list)
            or not page_numbers
            or any(
                isinstance(page_number, bool)
                or not isinstance(page_number, int)
                or page_number < 1
                for page_number in page_numbers
            )
        ):
            raise ValueError(
                f"Chunk at position {position} must have positive integer page_numbers."
            )

        for field, expected in (
            ("start_page", page_numbers[0]),
            ("end_page", page_numbers[-1]),
        ):
            page_number = chunk.get(field)
            if (
                isinstance(page_number, bool)
                or not isinstance(page_number, int)
                or page_number != expected
            ):
                raise ValueError(
                    f"Chunk at position {position} has inconsistent {field} metadata."
                )

    def _validate_vectors(
        self, vectors: object, expected_count: int, dimension: int, batch_index: int
    ) -> list[list[float]]:
        if not isinstance(vectors, list) or len(vectors) != expected_count:
            raise ValueError(
                f"Embedding provider batch {batch_index} returned "
                f"{len(vectors) if isinstance(vectors, list) else 'an invalid number of'} "
                f"vectors; expected {expected_count}."
            )

        validated: list[list[float]] = []
        for vector_index, vector in enumerate(vectors):
            if not isinstance(vector, list) or len(vector) != dimension:
                raise ValueError(
                    f"Embedding provider batch {batch_index}, vector {vector_index} "
                    f"must contain exactly {dimension} values."
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
                for value in vector
            ):
                raise ValueError(
                    f"Embedding provider batch {batch_index}, vector {vector_index} "
                    "contains a non-finite or non-numeric value."
                )

            norm = hypot(*(float(value) for value in vector))
            if not isclose(
                norm,
                1.0,
                rel_tol=_NORMALIZATION_TOLERANCE,
                abs_tol=_NORMALIZATION_TOLERANCE,
            ):
                raise ValueError(
                    f"Embedding provider batch {batch_index}, vector {vector_index} "
                    "is not L2-normalized."
                )
            validated.append(vector)
        return validated

    def embed_chunks(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        for position, chunk in enumerate(chunks):
            self._validate_chunk(chunk, position)

        dimension = self.provider.dimension
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError("Embedding provider dimension must be a positive integer.")

        embedded_chunks: list[EmbeddedChunk] = []
        for batch_index, batch_start in enumerate(
            range(0, len(chunks), self.batch_size)
        ):
            batch = chunks[batch_start : batch_start + self.batch_size]
            vectors = self._validate_vectors(
                self.provider.embed_texts([chunk["text"] for chunk in batch]),
                len(batch),
                dimension,
                batch_index,
            )
            for chunk, vector in zip(batch, vectors):
                embedded_chunks.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "document_id": chunk["document_id"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "page_numbers": list(chunk["page_numbers"]),
                        "start_page": chunk["start_page"],
                        "end_page": chunk["end_page"],
                        "embedding": list(vector),
                    }
                )
        return embedded_chunks

    def embed_query(self, query: str) -> list[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty.")

        dimension = self.provider.dimension
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError("Embedding provider dimension must be a positive integer.")

        vectors = self._validate_vectors(
            self.provider.embed_texts([query.strip()]),
            expected_count=1,
            dimension=dimension,
            batch_index=0,
        )
        return list(vectors[0])
