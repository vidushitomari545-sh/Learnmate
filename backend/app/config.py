import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIRECTORY = Path(__file__).resolve().parents[1]
ENV_FILE = BACKEND_DIRECTORY / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=False)

DATA_DIRECTORY = BACKEND_DIRECTORY / "data"
DATABASE_PATH = DATA_DIRECTORY / "learnmate.sqlite3"
DOCUMENT_STORAGE_ROOT = Path(
    os.environ.get(
        "LEARNMATE_DOCUMENT_STORAGE",
        str(DATA_DIRECTORY / "documents"),
    )
)
DEVELOPMENT_OWNER_ID = os.environ.get(
    "LEARNMATE_DEVELOPMENT_OWNER_ID",
    "local-development",
)


def _positive_integer_setting(environment_name: str, default: int) -> int:
    value = os.environ.get(environment_name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{environment_name} must be a positive integer.") from error
    if parsed < 1:
        raise ValueError(f"{environment_name} must be a positive integer.")
    return parsed


def _nonnegative_integer_setting(environment_name: str, default: int) -> int:
    value = os.environ.get(environment_name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{environment_name} must be a non-negative integer.") from error
    if parsed < 0:
        raise ValueError(f"{environment_name} must be a non-negative integer.")
    return parsed


EMBEDDING_MODEL_NAME = os.environ.get(
    "LEARNMATE_EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
).strip()
EMBEDDING_DEVICE = os.environ.get("LEARNMATE_EMBEDDING_DEVICE", "cpu").strip()
EMBEDDING_DIMENSION = _positive_integer_setting(
    "LEARNMATE_EMBEDDING_DIMENSION",
    384,
)
EMBEDDING_BATCH_SIZE = _positive_integer_setting(
    "LEARNMATE_EMBEDDING_BATCH_SIZE",
    32,
)
RAG_MAX_CONTEXT_TOKENS = _positive_integer_setting(
    "LEARNMATE_RAG_MAX_CONTEXT_TOKENS",
    6000,
)
RAG_MAX_CONTEXT_BYTES = _positive_integer_setting(
    "LEARNMATE_RAG_MAX_CONTEXT_BYTES",
    24000,
)
RAG_MAX_OUTPUT_TOKENS = _positive_integer_setting(
    "LEARNMATE_RAG_MAX_OUTPUT_TOKENS",
    800,
)
RAG_MAX_QUESTION_CHARACTERS = _positive_integer_setting(
    "LEARNMATE_RAG_MAX_QUESTION_CHARACTERS",
    4000,
)
RAG_TOP_K = _positive_integer_setting("LEARNMATE_RAG_TOP_K", 5)
DOCUMENT_SUMMARY_MAX_OUTPUT_TOKENS = _positive_integer_setting(
    "LEARNMATE_DOCUMENT_SUMMARY_MAX_OUTPUT_TOKENS",
    1600,
)

LLM_PROVIDER = os.environ.get("LEARNMATE_LLM_PROVIDER", "ollama").strip().lower()
LLM_MODEL = os.environ.get("LEARNMATE_LLM_MODEL", "gpt-6-luna").strip()
GEMINI_MODEL = os.environ.get(
    "LEARNMATE_GEMINI_MODEL",
    "gemini-3.6-flash",
).strip()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
OLLAMA_BASE_URL = os.environ.get(
    "LEARNMATE_OLLAMA_BASE_URL",
    "http://localhost:11434",
).strip()
OLLAMA_MODEL = os.environ.get("LEARNMATE_OLLAMA_MODEL", "llama3.2:3b").strip()
LLM_TIMEOUT_SECONDS = _positive_integer_setting(
    "LEARNMATE_LLM_TIMEOUT_SECONDS",
    20,
)
LLM_MAX_RETRIES = _nonnegative_integer_setting(
    "LEARNMATE_LLM_MAX_RETRIES",
    1,
)

if not EMBEDDING_MODEL_NAME:
    raise ValueError("LEARNMATE_EMBEDDING_MODEL_NAME must not be empty.")
if not EMBEDDING_DEVICE:
    raise ValueError("LEARNMATE_EMBEDDING_DEVICE must not be empty.")
if LLM_PROVIDER not in {"fake", "openai", "gemini", "ollama"}:
    raise ValueError(
        "LEARNMATE_LLM_PROVIDER must be 'fake', 'openai', 'gemini', or 'ollama'."
    )
if not LLM_MODEL:
    raise ValueError("LEARNMATE_LLM_MODEL must not be empty.")
if LLM_PROVIDER == "gemini":
    if not GEMINI_MODEL:
        raise ValueError("LEARNMATE_GEMINI_MODEL must not be empty.")
    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is required when the Gemini provider is selected."
        )
if LLM_PROVIDER == "ollama":
    if not OLLAMA_BASE_URL:
        raise ValueError("LEARNMATE_OLLAMA_BASE_URL must not be empty.")
    if not OLLAMA_MODEL:
        raise ValueError("LEARNMATE_OLLAMA_MODEL must not be empty.")
