"""Provider-neutral language model request, response, and error contracts."""

from dataclasses import dataclass
from typing import AbstractSet, Literal, Protocol


@dataclass(frozen=True)
class LLMContextSource:
    source_id: str
    text: str


@dataclass(frozen=True)
class LLMRequest:
    question: str
    system_prompt: str
    user_prompt: str
    sources: tuple[LLMContextSource, ...]
    max_output_tokens: int
    require_all_source_ids: bool = False


@dataclass(frozen=True)
class LLMResponse:
    answer: str
    source_ids: tuple[str, ...]
    insufficient_evidence: bool


SourceIdsContainerType = Literal["dict", "list", "NoneType", "other", "str", "tuple"]
_SOURCE_IDS_CONTAINER_TYPES: dict[str, SourceIdsContainerType] = {
    "dict": "dict",
    "list": "list",
    "NoneType": "NoneType",
    "str": "str",
    "tuple": "tuple",
}


@dataclass(frozen=True)
class SourceIdsShapeDiagnostic:
    container_type: SourceIdsContainerType
    item_count: int | None
    non_string_member_count: int
    empty_string_member_count: int
    duplicate_string_member_count: int
    unknown_id_count: int | None
    is_empty: bool
    normalization_ran: bool

    def __post_init__(self) -> None:
        counts = (
            self.non_string_member_count,
            self.empty_string_member_count,
            self.duplicate_string_member_count,
        )
        if (
            self.container_type not in _SOURCE_IDS_CONTAINER_TYPES.values()
            and self.container_type != "other"
        ) or (
            self.item_count is not None
            and (type(self.item_count) is not int or self.item_count < 0)
        ) or any(type(count) is not int or count < 0 for count in counts) or (
            self.unknown_id_count is not None
            and (
                type(self.unknown_id_count) is not int
                or self.unknown_id_count < 0
            )
        ) or (
            type(self.is_empty) is not bool
            or type(self.normalization_ran) is not bool
        ):
            raise ValueError("Invalid source-ID shape diagnostic.")


def inspect_source_ids_shape(
    value: object,
    expected_source_ids: AbstractSet[str] | None,
    *,
    normalization_ran: bool,
) -> SourceIdsShapeDiagnostic:
    type_name = type(value).__name__
    container_type = _SOURCE_IDS_CONTAINER_TYPES.get(type_name, "other")
    if not isinstance(value, (list, tuple)):
        return SourceIdsShapeDiagnostic(
            container_type=container_type,
            item_count=None,
            non_string_member_count=0,
            empty_string_member_count=0,
            duplicate_string_member_count=0,
            unknown_id_count=None,
            is_empty=False,
            normalization_ran=normalization_ran,
        )

    non_string_count = 0
    empty_string_count = 0
    duplicate_string_count = 0
    unknown_id_count = 0
    seen: set[str] = set()
    for member in value:
        if not isinstance(member, str):
            non_string_count += 1
            continue
        if not member.strip():
            empty_string_count += 1
        if member in seen:
            duplicate_string_count += 1
        else:
            seen.add(member)
        if (
            member.strip()
            and expected_source_ids is not None
            and member not in expected_source_ids
        ):
            unknown_id_count += 1

    return SourceIdsShapeDiagnostic(
        container_type=container_type,
        item_count=len(value),
        non_string_member_count=non_string_count,
        empty_string_member_count=empty_string_count,
        duplicate_string_member_count=duplicate_string_count,
        unknown_id_count=(
            unknown_id_count if expected_source_ids is not None else None
        ),
        is_empty=not value,
        normalization_ran=normalization_ran,
    )


class LLMProviderError(RuntimeError):
    """Base class for failures returned by a language model provider."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.summary_diagnostic: object | None = None


class LLMTimeoutError(LLMProviderError):
    """The provider did not respond before the request timeout."""


class LLMRateLimitError(LLMProviderError):
    """The provider rejected the request due to a rate or quota limit."""


class LLMTransientError(LLMProviderError):
    """A temporary provider or network failure prevented generation."""


class LLMAuthenticationError(LLMProviderError):
    """The provider rejected the configured credentials."""


class LLMMalformedResponseError(LLMProviderError):
    """The provider response did not satisfy the structured output contract."""

    def __init__(
        self,
        message: str,
        *,
        diagnostic_category: str = "field_type",
        actual_response_type: str = "unknown",
        source_ids_rejected: bool = False,
        source_ids_shape: SourceIdsShapeDiagnostic | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostic_category = diagnostic_category
        self.actual_response_type = actual_response_type
        self.source_ids_rejected = source_ids_rejected
        self.source_ids_shape = source_ids_shape


class LLMProvider(Protocol):
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a grounded answer for the request."""
        ...
