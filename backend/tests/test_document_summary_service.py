import json
from math import ceil

import pytest

from app.services.document_summary import (
    DocumentSummaryService,
    InvalidSummaryInputError,
    MalformedSummaryOutputError,
    NoExtractableTextError,
    SummaryContextLimitError,
)
from app.services.llm.interfaces import LLMRequest, LLMResponse


class RecordingSummaryProvider:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.source_id_override: tuple[str, ...] | None = None
        self.final_answer_override: str | None = None
        self.expand_map_output = False

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        sources = json.loads(request.user_prompt[request.user_prompt.index('[{"source_id"') :])
        source_ids = tuple(source["source_id"] for source in sources)
        if self.source_id_override is not None:
            source_ids = self.source_id_override

        if request.question == "Write the final exam-oriented document summary.":
            answer = self.final_answer_override or (
                "## Topic-wise explanations\n"
                "### Core topic\nA source-grounded explanation.\n\n"
                "## Definitions\nA definition from the source.\n\n"
                "## Examples\nA supported example.\n\n"
                "## Important concepts\nA supported concept.\n\n"
                "## Key takeaways\n- A supported takeaway.\n\n"
                "## Final revision\n- Recall the source's core idea."
            )
        elif request.question == "Combine the supplied section summaries.":
            answer = "### Combined notes\nPreserve each source detail."
        else:
            answer = (
                "### Section notes\n"
                + "Preserve a source-backed detail. " * 25
                if self.expand_map_output
                else "### Section notes\nPreserve terminology and examples."
            )

        return LLMResponse(answer, source_ids, False)


def pages_with_text(page_count: int, text_size: int = 1200) -> list[dict[str, object]]:
    return [
        {"page_number": page_number, "text": f"Page {page_number} " + "x" * text_size}
        for page_number in range(1, page_count + 1)
    ]


def parse_sources(request: LLMRequest) -> list[dict[str, object]]:
    return json.loads(request.user_prompt[request.user_prompt.index('[{"source_id"') :])


def test_summarizes_all_text_pages_and_returns_verified_page_references() -> None:
    provider = RecordingSummaryProvider()
    pages = pages_with_text(3)
    pages.append({"page_number": 4, "text": ""})
    progress: list[tuple[str, int, int]] = []

    result = DocumentSummaryService(provider).summarize_pages(
        "document-1",
        pages,
        page_count=4,
        text_status="partial_text",
        progress=lambda phase, current, total: progress.append(
            (phase, current, total)
        ),
    )

    assert all(request.require_all_source_ids for request in provider.requests)
    assert all(
        "set insufficient_evidence to false" in request.system_prompt
        for request in provider.requests
    )
    mapped_page_numbers = {
        page_number
        for request in provider.requests
        if request.question == "Summarize the supplied PDF section."
        for source in parse_sources(request)
        for page_number in source["page_numbers"]
    }
    assert mapped_page_numbers == {1, 2, 3}
    assert result.page_references == (1, 2, 3)
    assert result.pages_without_text == (4,)
    assert result.chunk_count >= 3
    assert result.text_status == "partial_text"
    assert "## Final revision" in result.summary_markdown
    assert progress[0][0] == "mapping"
    assert progress[-1] == ("finalizing", 1, 1)


def test_large_document_uses_bounded_hierarchical_groups_without_omission() -> None:
    provider = RecordingSummaryProvider()
    provider.expand_map_output = True
    pages = pages_with_text(12, text_size=1800)
    progress: list[tuple[str, int, int]] = []
    service = DocumentSummaryService(
        provider,
        max_context_tokens=1200,
        max_context_bytes=5000,
        max_output_tokens=200,
    )

    result = service.summarize_pages(
        "large-document",
        pages,
        page_count=len(pages),
        text_status="text_found",
        progress=lambda phase, current, total: progress.append(
            (phase, current, total)
        ),
    )

    map_sources = [
        source
        for request in provider.requests
        if request.question == "Summarize the supplied PDF section."
        for source in parse_sources(request)
    ]
    assert len(map_sources) == result.chunk_count
    assert {
        page_number
        for source in map_sources
        for page_number in source["page_numbers"]
    } == set(range(1, 13))
    assert any(request.question == "Combine the supplied section summaries." for request in provider.requests)
    assert result.page_references == tuple(range(1, 13))

    for request in provider.requests:
        full_prompt = f"{request.system_prompt}\n{request.user_prompt}"
        assert (
            ceil(len(full_prompt) / 4) + request.max_output_tokens + 256
            <= service.max_context_tokens
        )
        assert len(full_prompt.encode("utf-8")) <= service.max_context_bytes
    assert any(phase == "reducing" for phase, _, _ in progress)


def test_rejects_missing_source_ids_instead_of_returning_partial_summary() -> None:
    provider = RecordingSummaryProvider()
    provider.source_id_override = ()

    with pytest.raises(MalformedSummaryOutputError, match="source"):
        DocumentSummaryService(provider).summarize_pages(
            "document-1",
            pages_with_text(1),
            page_count=1,
            text_status="text_found",
        )


def test_rejects_final_summary_without_required_exam_sections() -> None:
    provider = RecordingSummaryProvider()
    provider.final_answer_override = "A paragraph without required headings."

    with pytest.raises(MalformedSummaryOutputError, match="required"):
        DocumentSummaryService(provider).summarize_pages(
            "document-1",
            pages_with_text(1),
            page_count=1,
            text_status="text_found",
        )


def test_rejects_incomplete_page_records_without_summarizing_partial_input() -> None:
    provider = RecordingSummaryProvider()

    with pytest.raises(InvalidSummaryInputError, match="page count"):
        DocumentSummaryService(provider).summarize_pages(
            "document-1",
            [{"page_number": 1, "text": "Text"}],
            page_count=2,
            text_status="text_found",
        )
    assert provider.requests == []


def test_no_extractable_text_is_reported_without_calling_provider() -> None:
    provider = RecordingSummaryProvider()

    with pytest.raises(NoExtractableTextError):
        DocumentSummaryService(provider).summarize_pages(
            "scanned-document",
            [{"page_number": 1, "text": "  "}],
            page_count=1,
            text_status="no_text_found",
        )
    assert provider.requests == []


def test_oversized_source_is_rejected_without_truncating_or_calling_provider() -> None:
    provider = RecordingSummaryProvider()
    service = DocumentSummaryService(
        provider,
        max_context_tokens=300,
        max_context_bytes=1000,
        max_output_tokens=20,
    )

    with pytest.raises(SummaryContextLimitError, match="complete summary source"):
        service.summarize_pages(
            "document-1",
            pages_with_text(1),
            page_count=1,
            text_status="text_found",
        )
    assert provider.requests == []
