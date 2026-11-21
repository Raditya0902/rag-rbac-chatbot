"""Chat history must be private to the user who created it.

History is stored in Redis under the session id. If the client-supplied id
were used as the key directly, any user could send another user's session id
(the frontend used the guessable "<username>_session") and get that user's
conversation, answers from other departments included, loaded into their prompt.
"""
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import chat
from src.api.limiter import limiter


def _client_as(username: str, role: str) -> TestClient:
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(chat.router)
    app.dependency_overrides[chat._get_user] = lambda: {"username": username, "role": role}
    return TestClient(app)


def _fake_run_query(query, namespaces, session_id=None, llm=None):
    return {"answer": "ok", "session_id": session_id, "guardrail_blocked": False}


@pytest.fixture(autouse=True)
def _no_rate_limit():
    limiter.enabled = False
    yield
    limiter.enabled = True


def _history_key_used(username: str, role: str, session_id: str | None) -> tuple[str, dict]:
    with patch.object(chat, "run_query", side_effect=_fake_run_query) as run_query:
        body = {"message": "hi"} if session_id is None else {"message": "hi", "session_id": session_id}
        response = _client_as(username, role).post("/chat", json=body)
    assert response.status_code == 200
    return run_query.call_args.args[2], response.json()


def test_same_session_id_from_two_users_maps_to_different_histories():
    sam_key, _ = _history_key_used("sam", "Finance", "sam_session")
    bruce_key, _ = _history_key_used("bruce", "Marketing", "sam_session")

    assert sam_key != bruce_key


def test_session_key_is_stable_for_the_same_user():
    first, _ = _history_key_used("sam", "Finance", "abc")
    second, _ = _history_key_used("sam", "Finance", "abc")

    assert first == second


def test_client_gets_back_the_session_id_it_sent():
    _, body = _history_key_used("sam", "Finance", "abc")

    assert body["session_id"] == "abc"


def test_new_session_id_is_returned_and_reusable():
    first_key, body = _history_key_used("sam", "Finance", None)
    again_key, _ = _history_key_used("sam", "Finance", body["session_id"])

    assert first_key == again_key


def test_stream_endpoint_scopes_session_to_user():
    def fake_stream(query, namespaces, session_id=None):
        yield "ok"

    keys = []
    for username, role in [("sam", "Finance"), ("bruce", "Marketing")]:
        with patch.object(chat, "stream_query", side_effect=fake_stream) as stream_query:
            _client_as(username, role).post("/chat/stream", json={"message": "hi", "session_id": "shared"})
        keys.append(stream_query.call_args.args[2])

    assert keys[0] != keys[1]
