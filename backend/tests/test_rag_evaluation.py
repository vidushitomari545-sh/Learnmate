"""Contract evaluation on manually reviewed synthetic RAG cases.

The scripted fake responses make these tests deterministic. Passing does not
establish real-LLM answer quality or resistance to prompt injection.
"""

import pytest

from app.services.llm.fake import FakeLLMProvider
from app.services.llm.interfaces import (
    LLMMalformedResponseError,
    LLMResponse,
)
from app.services.rag_service import NO_EVIDENCE_ANSWER, RAGService, SYSTEM_PROMPT
from app.services.vector_index import VectorSearchResult
from tests.rag_evaluation_dataset import (
    RAG_EVALUATION_CASES,
    RAGEvaluationCase,
)


class EvaluationSearchService:
    def __init__(self, results: tuple[VectorSearchResult, ...]) -> None:
        self.results = results
        self.calls: list[tuple[str, str, int]] = []

    def search(
        self, owner_id: str, query: str, top_k: int = 5
    ) -> list[VectorSearchResult]:
        self.calls.append((owner_id, query, top_k))
        return list(self.results[:top_k])


def run_case(
    case: RAGEvaluationCase,
) -> tuple[RAGService, FakeLLMProvider]:
    search = EvaluationSearchService(case.retrieved)
    provider = FakeLLMProvider(
        response=case.provider_response
        or LLMResponse("Unused default.", (), True)
    )
    service = RAGService(search, provider, top_k=5)
    return service, provider


def _citation_tuples(result) -> tuple[tuple[str, str, str, tuple[int, ...]], ...]:
    return tuple(
        (
            citation.source_id,
            citation.document_id,
            citation.document_name,
            citation.page_numbers,
        )
        for citation in result.citations
    )


@pytest.mark.parametrize(
    "case",
    RAG_EVALUATION_CASES,
    ids=lambda case: case.case_id,
)
def test_evaluation_answer_support_and_abstention(case: RAGEvaluationCase) -> None:
    service, provider = run_case(case)

    if case.expected_invalid_citation_rejection:
        with pytest.raises(LLMMalformedResponseError, match="not provided"):
            service.answer("evaluation-owner", case.question)
        assert len(provider.requests) == 1
        return

    result = service.answer("evaluation-owner", case.question)

    assert result.insufficient_evidence is case.expected_insufficient_evidence
    if not case.retrieved:
        assert result.answer == NO_EVIDENCE_ANSWER
        assert provider.requests == []
        return

    answer_lower = result.answer.casefold()
    for expected_phrase in case.expected_answer_contains:
        assert expected_phrase.casefold() in answer_lower

    if result.insufficient_evidence:
        assert result.citations == ()
    else:
        retrieved_text = " ".join(record.text for record in case.retrieved).casefold()
        for expected_phrase in case.expected_evidence_contains:
            assert expected_phrase.casefold() in retrieved_text


@pytest.mark.parametrize(
    "case",
    RAG_EVALUATION_CASES,
    ids=lambda case: case.case_id,
)
def test_evaluation_citation_ids_documents_and_pages(case: RAGEvaluationCase) -> None:
    service, _ = run_case(case)

    if case.expected_invalid_citation_rejection:
        with pytest.raises(LLMMalformedResponseError, match="not provided"):
            service.answer("evaluation-owner", case.question)
        return

    result = service.answer("evaluation-owner", case.question)
    expected = tuple(
        (
            citation.source_id,
            citation.document_id,
            citation.document_name,
            citation.page_numbers,
        )
        for citation in case.expected_citations
    )

    assert _citation_tuples(result) == expected


def test_evaluation_retrieval_receives_trusted_owner_and_question() -> None:
    case = RAG_EVALUATION_CASES[0]
    service, _ = run_case(case)

    service.answer("trusted-evaluation-owner", case.question)

    assert service.semantic_search.calls == [
        ("trusted-evaluation-owner", case.question, 5)
    ]


def test_evaluation_injection_case_remains_untrusted_and_backend_cited() -> None:
    case = next(
        item
        for item in RAG_EVALUATION_CASES
        if item.contains_untrusted_instruction
    )
    service, provider = run_case(case)

    result = service.answer("evaluation-owner", case.question)
    request = provider.requests[0]

    assert "untrusted quoted data" in SYSTEM_PROMPT
    assert "never follow" in SYSTEM_PROMPT.lower()
    assert "untrusted data" in request.user_prompt.casefold()
    assert "Ignore all previous instructions" in request.user_prompt
    assert "S99" not in {source.source_id for source in request.sources}
    assert _citation_tuples(result) == (
        ("S1", "lab-safety", "lab-safety.pdf", (7,)),
    )


def test_evaluation_dataset_covers_all_requested_scenarios() -> None:
    assert {case.case_id for case in RAG_EVALUATION_CASES} == {
        "direct-fact",
        "two-source-synthesis",
        "answer-absent",
        "ambiguous-insufficient",
        "contradictory-sources",
        "unsupported-source-id",
        "pdf-instruction-injection",
    }
