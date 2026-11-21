import secrets

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GROQ_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_PROJECT: str = "rag-rbac-chatbot"

    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHROMA_PERSIST_DIR: str = "./vectorstore"
    RETRIEVAL_TOP_K: int = 8      # final docs passed to LLM after re-ranking
    RETRIEVAL_FETCH_K: int = 24   # candidates fetched per retriever before re-ranking
    LLM_MODEL: str = "openai/gpt-oss-120b"
    # RAGAS judge: provider is "groq" or "deepseek". An empty model means the
    # provider's default (groq: openai/gpt-oss-20b, deepseek: deepseek-chat).
    EVAL_JUDGE_PROVIDER: str = "groq"
    EVAL_JUDGE_MODEL: str = ""
    DEEPSEEK_API_KEY: str = ""

    ENABLE_GUARD_MODEL: bool = False
    GUARD_MODEL: str = "meta-llama/llama-prompt-guard-2-86m"
    GUARD_THRESHOLD: float = 0.5  # injection probability at or above which a query is blocked
    REDIS_URL: str = "redis://localhost:6379"

    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60


settings = Settings()

# Fall back to a random key if not set — set JWT_SECRET_KEY in .env for production
if not settings.JWT_SECRET_KEY:
    settings.JWT_SECRET_KEY = secrets.token_hex(32)
