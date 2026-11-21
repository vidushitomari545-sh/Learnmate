"""Deterministic fake provider for tests and the provider-independent stage."""

from app.services.llm.interfaces import LLMRequest, LLMResponse


class FakeLLMProvider:
    def __init__(
        self,
        response: LLMResponse | None = None,
    ) -> None:
        self.response = response or LLMResponse(
            answer="I couldn't determine the answer from the provided sources.",
            source_ids=(),
            insufficient_evidence=True,
        )
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.response
