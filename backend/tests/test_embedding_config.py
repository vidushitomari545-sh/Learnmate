import importlib
from pathlib import Path

import dotenv
import pytest
from dotenv import load_dotenv as original_load_dotenv

from app import config

EMBEDDING_ENVIRONMENT_KEYS = (
    "LEARNMATE_EMBEDDING_MODEL_NAME",
    "LEARNMATE_EMBEDDING_DEVICE",
    "LEARNMATE_EMBEDDING_DIMENSION",
    "LEARNMATE_EMBEDDING_BATCH_SIZE",
    "LEARNMATE_RAG_MAX_CONTEXT_TOKENS",
    "LEARNMATE_RAG_MAX_OUTPUT_TOKENS",
    "LEARNMATE_RAG_MAX_QUESTION_CHARACTERS",
    "LEARNMATE_RAG_TOP_K",
    "LEARNMATE_RAG_MAX_CONTEXT_BYTES",
    "LEARNMATE_LLM_PROVIDER",
    "LEARNMATE_LLM_MODEL",
    "LEARNMATE_GEMINI_MODEL",
    "LEARNMATE_OLLAMA_BASE_URL",
    "LEARNMATE_OLLAMA_MODEL",
    "LEARNMATE_LLM_TIMEOUT_SECONDS",
    "LEARNMATE_LLM_MAX_RETRIES",
    "GEMINI_API_KEY",
)


@pytest.fixture
def reload_config(monkeypatch: pytest.MonkeyPatch):
    for key in EMBEDDING_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda **_: False)
    importlib.reload(config)
    try:
        yield config
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_backend_dotenv_is_loaded_from_configured_location(
    reload_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LEARNMATE_LLM_PROVIDER=openai\n"
        "LEARNMATE_LLM_MODEL=gpt-6-luna\n"
        "LEARNMATE_GEMINI_MODEL=gemini-3.6-flash\n"
        "GEMINI_API_KEY=test-only-placeholder\n"
        "OPENAI_API_KEY=test-only-placeholder\n",
        encoding="utf-8",
    )
    expected_env_file = reload_config.ENV_FILE
    calls: list[tuple[Path, bool]] = []
    def load_test_environment(dotenv_path: Path, override: bool) -> bool:
        calls.append((Path(dotenv_path), override))
        return original_load_dotenv(dotenv_path=env_file, override=override)

    monkeypatch.setattr(dotenv, "load_dotenv", load_test_environment)
    loaded_config = importlib.reload(reload_config)

    assert loaded_config.LLM_PROVIDER == "openai"
    assert loaded_config.LLM_MODEL == "gpt-6-luna"
    assert loaded_config.GEMINI_MODEL == "gemini-3.6-flash"
    assert bool(loaded_config.GEMINI_API_KEY)
    assert calls == [(expected_env_file, False)]


def test_process_environment_takes_precedence_over_dotenv(
    reload_config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LEARNMATE_LLM_PROVIDER=openai\n"
        "LEARNMATE_LLM_MODEL=dotenv-model\n"
        "LEARNMATE_GEMINI_MODEL=dotenv-gemini-model\n"
        "GEMINI_API_KEY=dotenv-test-placeholder\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LEARNMATE_LLM_PROVIDER", "fake")
    monkeypatch.setenv("LEARNMATE_LLM_MODEL", "process-model")
    monkeypatch.setenv("LEARNMATE_GEMINI_MODEL", "process-gemini-model")
    monkeypatch.setenv("GEMINI_API_KEY", "process-test-placeholder")
    def load_test_environment(dotenv_path: Path, override: bool) -> bool:
        return original_load_dotenv(dotenv_path=env_file, override=override)

    monkeypatch.setattr(dotenv, "load_dotenv", load_test_environment)
    loaded_config = importlib.reload(reload_config)

    assert loaded_config.LLM_PROVIDER == "fake"
    assert loaded_config.LLM_MODEL == "process-model"
    assert loaded_config.GEMINI_MODEL == "process-gemini-model"
    assert loaded_config.GEMINI_API_KEY == "process-test-placeholder"


def test_embedding_settings_have_expected_defaults(reload_config) -> None:
    assert reload_config.EMBEDDING_MODEL_NAME == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert reload_config.EMBEDDING_DEVICE == "cpu"
    assert reload_config.EMBEDDING_DIMENSION == 384
    assert reload_config.EMBEDDING_BATCH_SIZE == 32
    assert reload_config.RAG_MAX_CONTEXT_TOKENS == 6000
    assert reload_config.RAG_MAX_OUTPUT_TOKENS == 800
    assert reload_config.RAG_MAX_QUESTION_CHARACTERS == 4000
    assert reload_config.RAG_TOP_K == 5
    assert reload_config.RAG_MAX_CONTEXT_BYTES == 24000
    assert reload_config.LLM_PROVIDER == "ollama"
    assert reload_config.LLM_MODEL == "gpt-6-luna"
    assert reload_config.GEMINI_MODEL == "gemini-3.6-flash"
    assert reload_config.GEMINI_API_KEY == ""
    assert reload_config.OLLAMA_BASE_URL == "http://localhost:11434"
    assert reload_config.OLLAMA_MODEL == "llama3.2:3b"
    assert reload_config.LLM_TIMEOUT_SECONDS == 20
    assert reload_config.LLM_MAX_RETRIES == 1


def test_rag_limits_read_custom_environment_values(
    reload_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEARNMATE_RAG_MAX_CONTEXT_TOKENS", "4096")
    monkeypatch.setenv("LEARNMATE_RAG_MAX_OUTPUT_TOKENS", "512")
    monkeypatch.setenv("LEARNMATE_RAG_MAX_QUESTION_CHARACTERS", "2000")
    monkeypatch.setenv("LEARNMATE_RAG_TOP_K", "7")
    monkeypatch.setenv("LEARNMATE_RAG_MAX_CONTEXT_BYTES", "16384")
    monkeypatch.setenv("LEARNMATE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("LEARNMATE_LLM_MODEL", "custom-model")
    monkeypatch.setenv("LEARNMATE_LLM_TIMEOUT_SECONDS", "35")
    monkeypatch.setenv("LEARNMATE_LLM_MAX_RETRIES", "0")

    custom_config = importlib.reload(reload_config)

    assert custom_config.RAG_MAX_CONTEXT_TOKENS == 4096
    assert custom_config.RAG_MAX_OUTPUT_TOKENS == 512
    assert custom_config.RAG_MAX_QUESTION_CHARACTERS == 2000
    assert custom_config.RAG_TOP_K == 7
    assert custom_config.RAG_MAX_CONTEXT_BYTES == 16384
    assert custom_config.LLM_PROVIDER == "openai"
    assert custom_config.LLM_MODEL == "custom-model"
    assert custom_config.LLM_TIMEOUT_SECONDS == 35
    assert custom_config.LLM_MAX_RETRIES == 0


def test_gemini_provider_uses_configured_model_and_key(
    reload_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEARNMATE_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LEARNMATE_GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-placeholder")

    gemini_config = importlib.reload(reload_config)

    assert gemini_config.LLM_PROVIDER == "gemini"
    assert gemini_config.GEMINI_MODEL == "gemini-test-model"
    assert bool(gemini_config.GEMINI_API_KEY)


def test_gemini_provider_requires_api_key(
    reload_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEARNMATE_LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GEMINI_API_KEY is required"):
        importlib.reload(reload_config)


def test_openai_configuration_does_not_require_gemini_key(
    reload_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEARNMATE_LLM_PROVIDER", "openai")
    monkeypatch.setenv("LEARNMATE_LLM_MODEL", "gpt-6-luna")
    monkeypatch.setenv("LEARNMATE_GEMINI_MODEL", "")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    openai_config = importlib.reload(reload_config)

    assert openai_config.LLM_PROVIDER == "openai"
    assert openai_config.LLM_MODEL == "gpt-6-luna"
    assert openai_config.GEMINI_API_KEY == ""


def test_ollama_provider_uses_configured_local_model_and_url(
    reload_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEARNMATE_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LEARNMATE_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.setenv("LEARNMATE_OLLAMA_MODEL", "custom-local-model")

    ollama_config = importlib.reload(reload_config)

    assert ollama_config.LLM_PROVIDER == "ollama"
    assert ollama_config.OLLAMA_BASE_URL == "http://127.0.0.1:11434"
    assert ollama_config.OLLAMA_MODEL == "custom-local-model"


@pytest.mark.parametrize(
    ("environment_name", "value"),
    [
        ("LEARNMATE_OLLAMA_BASE_URL", ""),
        ("LEARNMATE_OLLAMA_MODEL", " "),
    ],
)
def test_ollama_provider_requires_nonempty_settings(
    reload_config,
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    value: str,
) -> None:
    monkeypatch.setenv("LEARNMATE_LLM_PROVIDER", "ollama")
    monkeypatch.setenv(environment_name, value)

    with pytest.raises(ValueError, match=environment_name):
        importlib.reload(reload_config)


def test_gemini_provider_rejects_empty_model(
    reload_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEARNMATE_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LEARNMATE_GEMINI_MODEL", "")
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-placeholder")

    with pytest.raises(ValueError, match="LEARNMATE_GEMINI_MODEL"):
        importlib.reload(reload_config)


def test_embedding_settings_read_custom_environment_values(
    reload_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LEARNMATE_EMBEDDING_MODEL_NAME", "custom/model")
    monkeypatch.setenv("LEARNMATE_EMBEDDING_DEVICE", "cuda:0")
    monkeypatch.setenv("LEARNMATE_EMBEDDING_DIMENSION", "768")
    monkeypatch.setenv("LEARNMATE_EMBEDDING_BATCH_SIZE", "16")

    custom_config = importlib.reload(reload_config)

    assert custom_config.EMBEDDING_MODEL_NAME == "custom/model"
    assert custom_config.EMBEDDING_DEVICE == "cuda:0"
    assert custom_config.EMBEDDING_DIMENSION == 768
    assert custom_config.EMBEDDING_BATCH_SIZE == 16


@pytest.mark.parametrize(
    ("environment_name", "value"),
    [
        ("LEARNMATE_EMBEDDING_MODEL_NAME", " "),
        ("LEARNMATE_EMBEDDING_DEVICE", ""),
        ("LEARNMATE_EMBEDDING_DIMENSION", "not-an-integer"),
        ("LEARNMATE_EMBEDDING_DIMENSION", "0"),
        ("LEARNMATE_EMBEDDING_BATCH_SIZE", "-1"),
        ("LEARNMATE_RAG_MAX_CONTEXT_TOKENS", "0"),
        ("LEARNMATE_RAG_MAX_OUTPUT_TOKENS", "abc"),
        ("LEARNMATE_RAG_MAX_QUESTION_CHARACTERS", "-1"),
        ("LEARNMATE_RAG_TOP_K", "0"),
        ("LEARNMATE_RAG_MAX_CONTEXT_BYTES", "0"),
        ("LEARNMATE_LLM_PROVIDER", "unknown"),
        ("LEARNMATE_LLM_MODEL", " "),
        ("LEARNMATE_LLM_TIMEOUT_SECONDS", "0"),
        ("LEARNMATE_LLM_MAX_RETRIES", "-1"),
    ],
)
def test_rejects_invalid_embedding_settings(
    reload_config,
    monkeypatch: pytest.MonkeyPatch,
    environment_name: str,
    value: str,
) -> None:
    monkeypatch.setenv(environment_name, value)

    with pytest.raises(ValueError, match=environment_name):
        importlib.reload(reload_config)
