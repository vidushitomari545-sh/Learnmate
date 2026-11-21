from typing import Protocol


class EmbeddingProvider(Protocol):
    """Provider contract for generating normalized text embeddings."""

    model_identifier: str
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalized vector per input, in the same order.

        Every returned vector must have ``dimension`` elements. Empty input
        must return an empty list.
        """
        ...
