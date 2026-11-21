"""Character-based document text chunking with source-page metadata.

Chunk boundaries prefer paragraph breaks, then sentence and whitespace
boundaries, falling back to the configured character limit. Overlap is
measured in Python string characters rather than model tokens.
"""

import re
import uuid
from collections.abc import Mapping, Sequence
from typing import TypedDict

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

_SENTENCE_BOUNDARY = re.compile(r"""[.!?]["')\]]*[ \t]+""")
_WHITESPACE_BOUNDARY = re.compile(r"[ \t]+")


class TextChunk(TypedDict):
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    page_numbers: list[int]
    start_page: int
    end_page: int


def _preferred_boundary(
    text: str, start: int, target: int, minimum_end: int
) -> int | None:
    paragraph_start = text.rfind("\n\n", start, target)
    if paragraph_start >= 0:
        paragraph_end = paragraph_start + 2
        if minimum_end <= paragraph_end <= target:
            return paragraph_end

    sentence_end = None
    for match in _SENTENCE_BOUNDARY.finditer(text, start, target):
        if match.end() >= minimum_end:
            sentence_end = match.end()
    if sentence_end is not None:
        return sentence_end

    line_break = text.rfind("\n", start, target)
    if line_break >= 0 and line_break + 1 >= minimum_end:
        return line_break + 1

    whitespace_end = None
    for match in _WHITESPACE_BOUNDARY.finditer(text, start, target):
        if match.end() >= minimum_end:
            whitespace_end = match.end()
    return whitespace_end


def chunk_text_pages(
    document_id: str,
    pages: Sequence[Mapping[str, object]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Chunk cleaned page records and retain their original page references."""
    if not document_id.strip():
        raise ValueError("document_id must not be empty.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if (
        isinstance(overlap, bool)
        or not isinstance(overlap, int)
        or overlap < 0
        or overlap >= chunk_size
    ):
        raise ValueError("overlap must be an integer from 0 to chunk_size - 1.")

    parts: list[str] = []
    page_spans: list[tuple[int, int, int]] = []
    current_length = 0
    for page in pages:
        page_number = page.get("page_number")
        page_text = page.get("text")
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
        ):
            raise ValueError("Each page_number must be a positive integer.")
        if not isinstance(page_text, str):
            raise ValueError("Each page text must be a string.")

        page_text = page_text.strip()
        if not page_text:
            continue
        if parts:
            parts.append("\n\n")
            current_length += 2
        page_start = current_length
        parts.append(page_text)
        current_length += len(page_text)
        page_spans.append((page_start, current_length, page_number))

    text = "".join(parts)
    if not text:
        return []

    chunks: list[TextChunk] = []
    start = 0
    while start < len(text):
        target = min(start + chunk_size, len(text))
        end = target
        if target < len(text):
            minimum_end = start + max(overlap + 1, chunk_size // 2)
            end = _preferred_boundary(text, start, target, minimum_end) or target

        chunk_text = text[start:end]
        source_pages = [
            page_number
            for page_start, page_end, page_number in page_spans
            if page_start < end and page_end > start
        ]
        if chunk_text and source_pages:
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "text": chunk_text,
                    "chunk_index": len(chunks),
                    "page_numbers": source_pages,
                    "start_page": source_pages[0],
                    "end_page": source_pages[-1],
                }
            )

        if end == len(text):
            break
        start = end - overlap

    return chunks
