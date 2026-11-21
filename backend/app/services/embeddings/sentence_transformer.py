"""Lazy Sentence Transformers adapter for the LearnMate embedding contract."""

from collections.abc import Callable
from math import hypot, isclose, isfinite
from numbers import Real
from threading import Lock
from typing import Protocol

EXPECTED_EMBEDDING_DIMENSION = 384
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "cpu"
_NORMALIZATION_TOLERANCE = 1e-5


class _SentenceTransformerModel(Protocol):
    def encode(
        self, sentences: list[str], *, normalize_embeddings: bool
    ) -> object: ...


ModelFactory = Callable[[str, str], _SentenceTransformerModel]


def _load_sentence_transformer_model(
    model_name: str, device: str
) -> _SentenceTransformerModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


class SentenceTransformerProvider:
    dimension = EXPECTED_EMBEDDING_DIMENSION

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        model_factory: ModelFactory | None = None,
        dimension: int = EXPECTED_EMBEDDING_DIMENSION,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty.")
        if not device.strip():
            raise ValueError("device must not be empty.")
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError("dimension must be a positive integer.")

        self.model_name = model_name
        self.model_identifier = model_name
        self.device = device
        self.dimension = dimension
        self._model_factory = model_factory or _load_sentence_transformer_model
        self._model: _SentenceTransformerModel | None = None
        self._model_lock = Lock()

    def _get_model(self) -> _SentenceTransformerModel:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = self._model_factory(self.model_name, self.device)
        return self._model

    @staticmethod
    def _as_python_lists(encoded: object) -> object:
        tolist = getattr(encoded, "tolist", None)
        if callable(tolist):
            return tolist()
        return encoded

    def _validate_embeddings(self, encoded: object, text_count: int) -> list[list[float]]:
        vectors = self._as_python_lists(encoded)
        if not isinstance(vectors, list) or len(vectors) != text_count:
            actual_count = len(vectors) if isinstance(vectors, list) else "invalid"
            raise ValueError(
                f"Sentence Transformer returned {actual_count} vectors; "
                f"expected {text_count}."
            )

        result: list[list[float]] = []
        for vector_index, vector in enumerate(vectors):
            if not isinstance(vector, list) or len(vector) != self.dimension:
                actual_dimension = len(vector) if isinstance(vector, list) else "invalid"
                raise ValueError(
                    f"Sentence Transformer vector {vector_index} has dimension "
                    f"{actual_dimension}; expected {self.dimension}."
                )

            python_vector: list[float] = []
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError(
                        f"Sentence Transformer vector {vector_index} contains "
                        "a non-numeric value."
                    )
                numeric_value = float(value)
                if not isfinite(numeric_value):
                    raise ValueError(
                        f"Sentence Transformer vector {vector_index} contains "
                        "a non-finite value."
                    )
                python_vector.append(numeric_value)

            norm = hypot(*python_vector)
            if not isclose(
                norm,
                1.0,
                rel_tol=_NORMALIZATION_TOLERANCE,
                abs_tol=_NORMALIZATION_TOLERANCE,
            ):
                raise ValueError(
                    f"Sentence Transformer vector {vector_index} is not "
                    "L2-normalized."
                )
            result.append(python_vector)
        return result

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("Every input text must be a string.")

        encoded = self._get_model().encode(
            texts,
            normalize_embeddings=True,
        )
        return self._validate_embeddings(encoded, len(texts))
