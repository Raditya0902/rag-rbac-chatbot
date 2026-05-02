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
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    ENABLE_GUARD_MODEL: bool = False
    GUARD_MODEL: str = "meta-llama/llama-guard-3-8b"
    REDIS_URL: str = "redis://localhost:6379"

    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 60


settings = Settings()

# Fall back to a random key if not set — set JWT_SECRET_KEY in .env for production
if not settings.JWT_SECRET_KEY:
    settings.JWT_SECRET_KEY = secrets.token_hex(32)
