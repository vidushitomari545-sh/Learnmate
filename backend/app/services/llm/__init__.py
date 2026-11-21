"""Provider-independent language model contracts and test doubles."""

from app.services.llm.fake import FakeLLMProvider
from app.services.llm.interfaces import (
    LLMAuthenticationError,
    LLMContextSource,
    LLMMalformedResponseError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMTransientError,
)

__all__ = [
    "FakeLLMProvider",
    "LLMAuthenticationError",
    "LLMContextSource",
    "LLMMalformedResponseError",
    "LLMProvider",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMTransientError",
]
