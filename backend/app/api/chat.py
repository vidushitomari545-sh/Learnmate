import logging

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.chat import ChatCitation, ChatRequest, ChatResponse
from app.services.dependencies import get_development_owner_id, get_rag_service
from app.services.llm.interfaces import (
    LLMAuthenticationError,
    LLMMalformedResponseError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMTransientError,
)
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

_SAFE_PROVIDER_STATUSES = {
    "DEADLINE_EXCEEDED",
    "INTERNAL",
    "INVALID_ARGUMENT",
    "PERMISSION_DENIED",
    "RESOURCE_EXHAUSTED",
    "UNAUTHENTICATED",
    "UNAVAILABLE",
    "UNKNOWN",
}
_SAFE_ERROR_CATEGORIES = {
    "authentication",
    "malformed_response",
    "provider",
    "rate_limit",
    "timeout",
    "transient",
}
_SAFE_FAILURE_STAGES = {
    "GEMINI_RESPONSE_EXTRACTION",
    "GEMINI_FIELD_TYPE_VALIDATION",
    "GEMINI_JSON_DECODING",
    "GEMINI_JSON_PARSING",
    "GEMINI_SCHEMA_VALIDATION",
    "RAG_SOURCE_VALIDATION",
}
_SAFE_PROVIDER_EXCEPTION_CATEGORIES = {
    "api_authentication",
    "api_other",
    "api_rate_limit",
    "api_transient",
    "connection",
    "dns",
    "timeout",
    "tls",
    "unknown",
}


def _log_provider_failure(error: LLMProviderError) -> None:
    status_code = getattr(error, "http_status_code", None)
    if type(status_code) is not int or not 100 <= status_code <= 599:
        status_code = 0

    provider_status = getattr(error, "provider_status", None)
    if (
        not isinstance(provider_status, str)
        or provider_status not in _SAFE_PROVIDER_STATUSES
    ):
        provider_status = "unknown"

    error_category = getattr(error, "error_category", None)
    if (
        not isinstance(error_category, str)
        or error_category not in _SAFE_ERROR_CATEGORIES
    ):
        error_category = "unknown"

    exception_class = getattr(error, "provider_exception_class", None)
    if (
        not isinstance(exception_class, str)
        or not exception_class.isascii()
        or not exception_class.replace("_", "").isalnum()
        or len(exception_class) > 80
    ):
        exception_class = "unknown"

    attempt_count = getattr(error, "attempt_count", None)
    if type(attempt_count) is not int or not 1 <= attempt_count <= 1000:
        attempt_count = 0

    failure_stage = getattr(error, "failure_stage", None)
    if not isinstance(failure_stage, str) or failure_stage not in _SAFE_FAILURE_STAGES:
        failure_stage = "unknown"

    exception_category = getattr(error, "provider_exception_category", None)
    if (
        not isinstance(exception_category, str)
        or exception_category not in _SAFE_PROVIDER_EXCEPTION_CATEGORIES
    ):
        exception_category = "unknown"

    logger.warning(
        "Chat provider request failed: "
        "http_status_code=%d provider_status=%s error_category=%s "
        "provider_exception_class=%s attempt_count=%d failure_stage=%s "
        "provider_exception_category=%s",
        status_code,
        provider_status,
        error_category,
        exception_class,
        attempt_count,
        failure_stage,
        exception_category,
        extra={
            "http_status_code": status_code,
            "provider_status": provider_status,
            "error_category": error_category,
            "provider_exception_class": exception_class,
            "attempt_count": attempt_count,
            "failure_stage": failure_stage,
            "provider_exception_category": exception_category,
        },
    )


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    owner_id: str = Depends(get_development_owner_id),
    service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    try:
        result = service.answer(owner_id, request.question)
    except LLMTimeoutError as error:
        _log_provider_failure(error)
        raise HTTPException(
            status_code=504,
            detail="The answer service timed out. Please try again.",
        ) from error
    except LLMRateLimitError as error:
        _log_provider_failure(error)
        raise HTTPException(
            status_code=429,
            detail="The answer service is temporarily rate limited. Please try later.",
        ) from error
    except (
        LLMAuthenticationError,
        LLMTransientError,
    ) as error:
        _log_provider_failure(error)
        raise HTTPException(
            status_code=503,
            detail="The answer service is temporarily unavailable.",
        ) from error
    except LLMMalformedResponseError as error:
        _log_provider_failure(error)
        raise HTTPException(
            status_code=502,
            detail="The answer service returned an invalid response.",
        ) from error
    except LLMProviderError as error:
        _log_provider_failure(error)
        raise HTTPException(
            status_code=503,
            detail="The answer service is temporarily unavailable.",
        ) from error
    except Exception as error:
        logger.exception("Unexpected chat request failure.")
        raise HTTPException(
            status_code=500,
            detail="The question could not be processed.",
        ) from error

    return ChatResponse(
        answer=result.answer,
        insufficient_evidence=result.insufficient_evidence,
        citations=[
            ChatCitation(
                document_name=citation.document_name,
                page_numbers=list(citation.page_numbers),
            )
            for citation in result.citations
        ],
    )
