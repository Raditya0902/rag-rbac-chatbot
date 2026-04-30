from langsmith import Client

from src.config import settings


def get_langsmith_client() -> Client | None:
    if not settings.LANGCHAIN_API_KEY:
        return None
    return Client(api_key=settings.LANGCHAIN_API_KEY)


def log_feedback(run_id: str, score: float, comment: str = "") -> None:
    client = get_langsmith_client()
    if client:
        client.create_feedback(run_id, key="user_rating", score=score, comment=comment)
