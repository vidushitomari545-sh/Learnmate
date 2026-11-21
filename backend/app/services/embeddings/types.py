from typing import TypedDict


class EmbeddedChunk(TypedDict):
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    page_numbers: list[int]
    start_page: int
    end_page: int
    embedding: list[float]
