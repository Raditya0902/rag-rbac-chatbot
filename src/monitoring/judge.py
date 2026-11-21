"""Build the LLM that RAGAS uses as its judge.

Scores are only comparable between runs that used the same provider and model,
so ragas_eval records both alongside every result.
"""
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from src.config import settings

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Model used when a provider is chosen without naming a model.
_DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "deepseek": "deepseek-chat",
}


def resolve_judge(provider: str | None, model: str | None) -> tuple[str, str]:
    """Return (provider, model), falling back to settings, then provider defaults."""
    provider = (provider or settings.EVAL_JUDGE_PROVIDER).lower()
    if provider not in _DEFAULT_MODELS:
        known = ", ".join(sorted(_DEFAULT_MODELS))
        raise ValueError(f"Unknown judge provider '{provider}'. Expected one of: {known}")
    return provider, model or settings.EVAL_JUDGE_MODEL or _DEFAULT_MODELS[provider]


def _require_key(name: str) -> str:
    value = getattr(settings, name)
    if not value:
        raise ValueError(f"{name} is not set. Add it to .env to use this judge provider.")
    return value


def build_judge_chat_model(provider: str, model: str) -> BaseChatModel:
    """Instantiate the chat model for an already-resolved provider and model."""
    if provider == "groq":
        return ChatGroq(model=model, api_key=_require_key("GROQ_API_KEY"))
    if provider == "deepseek":
        # DeepSeek exposes an OpenAI-compatible API.
        return ChatOpenAI(
            model=model,
            api_key=_require_key("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
        )
    raise ValueError(f"Unknown judge provider '{provider}'")
