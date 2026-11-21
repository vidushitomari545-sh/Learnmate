import json

import pytest

from app.services.document_summary import (
    DocumentSummaryService,
    MalformedSummaryOutputError,
)
from app.services.llm.interfaces import (
    LLMRequest,
    LLMResponse,
    LLMMalformedResponseError,
)
from app.services.llm.ollama_provider import OllamaProvider


def ollama_response(content: str) -> bytes:
    return json.dumps({"message": {"content": content}}).encode("utf-8")


def structured_response(
    answer: str,
    source_ids: tuple[str, ...],
    *,
    insufficient_evidence: bool = False,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "source_ids": list(source_ids),
            "insufficient_evidence": insufficient_evidence,
        }
    )


class SyntheticMapReduceProvider:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.responses: list[LLMResponse] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        source_ids = tuple(source.source_id for source in request.sources)
        if request.question == "Summarize the supplied PDF section.":
            answer = "### Section notes\n" + (
                "Preserve a supported detail. " * 25
            )
        elif request.question == "Combine the supplied section summaries.":
            answer = "### Combined notes\nPreserve supported details."
        else:
            answer = (
                "## Topic-wise explanations\n"
                "### Supported topic\nExplanation from the sources.\n\n"
                "## Final revision\n- Recall the supported idea."
            )
        response = LLMResponse(answer, source_ids, False)
        self.responses.append(response)
        return response


class InvalidSourceIdProvider:
    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse("### Notes\nA supported note.", ("NOT-A-SOURCE",), False)


class ListSourceIdProvider:
    def __init__(self, source_ids: list[object] | None = None) -> None:
        self.source_ids = source_ids

    def generate(self, request: LLMRequest) -> LLMResponse:
        source_ids: list[object] = (
            self.source_ids
            if self.source_ids is not None
            else [source.source_id for source in request.sources]
        )
        if request.question == "Write the final exam-oriented document summary.":
            answer = (
                "## Topic-wise explanations\n"
                "### Supported topic\nExplanation from the sources.\n\n"
                "## Final revision\n- Recall the supported idea."
            )
        else:
            answer = "### Notes\nA source-backed summary."
        return LLMResponse(answer, source_ids, False)  # type: ignore[arg-type]


def test_synthetic_valid_map_and_reduce_responses_preserve_supplied_source_ids() -> None:
    provider = SyntheticMapReduceProvider()
    pages = [
        {"page_number": number, "text": f"Page {number} " + "x" * 1800}
        for number in range(1, 13)
    ]
    service = DocumentSummaryService(
        provider,
        max_context_tokens=1200,
        max_context_bytes=5000,
        max_output_tokens=200,
    )

    result = service.summarize_pages(
        "synthetic-document",
        pages,
        page_count=len(pages),
        text_status="text_found",
    )

    mapping_requests = [
        request
        for request in provider.requests
        if request.question == "Summarize the supplied PDF section."
    ]
    reducing_requests = [
        request
        for request in provider.requests
        if request.question == "Combine the supplied section summaries."
    ]
    assert mapping_requests
    assert reducing_requests
    for request in [*mapping_requests, *reducing_requests]:
        assert request.sources
        assert len(request.sources) == len(
            {source.source_id for source in request.sources}
        )
    assert all(
        response.source_ids
        == tuple(source.source_id for source in request.sources)
        for request, response in zip(provider.requests, provider.responses)
    )
    assert result.page_references == tuple(range(1, 13))


@pytest.mark.parametrize(
    "answer",
    [
        "### Section notes\nA valid synthetic map answer.",
        "### Combined notes\nA valid synthetic reduce answer.",
    ],
    ids=["valid-map-answer", "valid-reduce-answer"],
)
def test_ollama_normalizes_valid_map_and_reduce_structured_json(
    answer: str,
) -> None:
    ids = ("C000001", "C000002") if "Combined" in answer else ("C000001",)
    result = OllamaProvider._parse_response(
        ollama_response(structured_response(answer, ids))
    )

    assert result == LLMResponse(answer, ids, False)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (
            "```json\n"
            + structured_response("### Notes\nWrapped output.", ("C000001",))
            + "\n```",
            "content was not valid JSON",
        ),
        (
            '{"answer":"### Notes\\nMissing IDs.","insufficient_evidence":false}',
            "required response fields",
        ),
        ("{not valid JSON", "content was not valid JSON"),
    ],
    ids=["markdown-wrapped-json", "missing-required-field", "malformed-json"],
)
def test_ollama_rejects_malformed_structured_response_fixtures(
    content: str, message: str
) -> None:
    with pytest.raises(LLMMalformedResponseError, match=message):
        OllamaProvider._parse_response(ollama_response(content))


def test_summary_service_rejects_unknown_source_ids_from_valid_json_response() -> None:
    pages = [{"page_number": 1, "text": "A source-backed statement."}]

    with pytest.raises(MalformedSummaryOutputError, match="source IDs"):
        DocumentSummaryService(InvalidSourceIdProvider()).summarize_pages(
            "synthetic-document",
            pages,
            page_count=1,
            text_status="text_found",
        )


def test_summary_normalizes_valid_list_source_ids_without_weakening_membership() -> None:
    provider = ListSourceIdProvider()
    result = DocumentSummaryService(provider).summarize_pages(
        "synthetic-document",
        [{"page_number": 1, "text": "A source-backed statement."}],
        page_count=1,
        text_status="text_found",
    )

    assert result.page_references == (1,)
    assert "## Final revision" in result.summary_markdown


@pytest.mark.parametrize(
    ("source_ids", "expected_actual_type"),
    [
        (["UNKNOWN-SYNTHETIC-ID"], "tuple"),
        ([17], "list"),
    ],
    ids=["unknown-id-is-still-rejected", "non-string-member-is-rejected"],
)
def test_summary_rejects_unknown_or_malformed_list_source_ids(
    source_ids: list[object],
    expected_actual_type: str,
) -> None:
    with pytest.raises(MalformedSummaryOutputError) as raised:
        DocumentSummaryService(ListSourceIdProvider(source_ids)).summarize_pages(
            "synthetic-document",
            [{"page_number": 1, "text": "A source-backed statement."}],
            page_count=1,
            text_status="text_found",
        )

    assert raised.value.diagnostic is not None
    assert raised.value.diagnostic.category == "source_ids"
    assert raised.value.diagnostic.actual_response_type == expected_actual_type
    assert raised.value.diagnostic.source_ids_rejected is True
    shape = raised.value.diagnostic.source_ids_shape
    assert shape is not None
    assert shape.container_type == "list"
    assert shape.item_count == 1
    assert shape.non_string_member_count == (1 if source_ids == [17] else 0)
    assert shape.unknown_id_count == (
        1 if source_ids == ["UNKNOWN-SYNTHETIC-ID"] else 0
    )
    assert shape.normalization_ran is True


def test_provider_parser_rejects_non_string_json_source_id_member() -> None:
    content = json.dumps(
        {
            "answer": "A supported note.",
            "source_ids": [17],
            "insufficient_evidence": False,
        }
    )

    with pytest.raises(LLMMalformedResponseError) as raised:
        OllamaProvider._parse_response(
            ollama_response(content),
            {"SYNTHETIC-SOURCE-ID"},
        )

    assert raised.value.diagnostic_category == "source_ids"
    assert raised.value.actual_response_type == "list"
    assert raised.value.source_ids_rejected is True
    shape = raised.value.source_ids_shape
    assert shape is not None
    assert shape.container_type == "list"
    assert shape.item_count == 1
    assert shape.non_string_member_count == 1
    assert shape.empty_string_member_count == 0
    assert shape.duplicate_string_member_count == 0
    assert shape.unknown_id_count == 0
    assert shape.is_empty is False
    assert shape.normalization_ran is False
