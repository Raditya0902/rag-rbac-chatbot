import os
import time
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


def api_login(username: str, password: str) -> dict:
    resp = requests.post(
        f"{API_URL}/auth/login",
        json={"username": username, "password": password},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def api_logout(token: str) -> None:
    try:
        requests.post(
            f"{API_URL}/auth/logout",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
        )
    except Exception:
        pass  # Best-effort; client clears state regardless


def api_feedback(token: str, run_id: str, score: int) -> None:
    requests.post(
        f"{API_URL}/feedback",
        json={"run_id": run_id, "score": score},
        headers={"Authorization": f"Bearer {token}"},
        timeout=5,
    )


def stream_chat(token: str, message: str, session_id: str):
    """Generator that yields answer tokens from the SSE stream.

    Captures __run_id__ and __sources__ markers into session state
    so the caller can access them after st.write_stream() returns.
    """
    with requests.post(
        f"{API_URL}/chat/stream",
        json={"message": message, "session_id": session_id},
        headers={"Authorization": f"Bearer {token}"},
        stream=True,
        timeout=60,
    ) as resp:
        resp.raise_for_status()
        sources = []
        run_id = None
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            if data.startswith("__run_id__:"):
                run_id = data[len("__run_id__:"):]
                continue
            if data.startswith("__sources__:"):
                payload = data[len("__sources__:"):]
                sources = [s for s in payload.split("|") if s]
                continue
            yield data
            time.sleep(0.02)  # 20ms per chunk — makes fast tokens visually stream
        st.session_state["_last_run_id"] = run_id
        st.session_state["_last_sources"] = sources


def login_page() -> None:
    st.title("Company Chatbot")
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        try:
            data = api_login(username, password)
            st.session_state.update({
                "logged_in": True,
                "username": data["username"],
                "role": data["role"],
                "token": data["access_token"],
                "session_id": str(uuid.uuid4()),
                "messages": [],
            })
            st.rerun()
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                st.error("Invalid credentials.")
            else:
                st.error(f"Login failed: {e.response.status_code}")
        except Exception as e:
            st.error(f"Could not connect to API: {e}")


def _render_feedback(msg: dict, idx: int) -> None:
    """Render thumbs up/down for an assistant message that hasn't been rated."""
    if msg.get("feedback") or not msg.get("run_id"):
        if msg.get("feedback") == "up":
            st.caption("👍 Feedback submitted")
        elif msg.get("feedback") == "down":
            st.caption("👎 Feedback submitted")
        return

    col1, col2, _ = st.columns([1, 1, 10])
    if col1.button("👍", key=f"up_{idx}"):
        api_feedback(st.session_state.token, msg["run_id"], 1)
        st.session_state.messages[idx]["feedback"] = "up"
        st.rerun()
    if col2.button("👎", key=f"down_{idx}"):
        api_feedback(st.session_state.token, msg["run_id"], 0)
        st.session_state.messages[idx]["feedback"] = "down"
        st.rerun()


def chat_page() -> None:
    st.title("Company Chatbot")
    st.caption(
        f"Logged in as **{st.session_state.username}** | Role: **{st.session_state.role}**"
    )
    if st.button("Logout"):
        api_logout(st.session_state.get("token", ""))
        st.session_state.clear()
        st.rerun()

    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.caption("Sources: " + ", ".join(msg["sources"]))
            if msg["role"] == "assistant":
                _render_feedback(msg, idx)

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                st.session_state["_last_run_id"] = None
                st.session_state["_last_sources"] = []
                answer = st.write_stream(
                    stream_chat(st.session_state.token, prompt, st.session_state.session_id)
                )
                run_id = st.session_state.pop("_last_run_id", None)
                sources = st.session_state.pop("_last_sources", [])
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
                msg = {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "run_id": run_id,
                    "feedback": None,
                }
                st.session_state.messages.append(msg)
                # render feedback buttons for the just-added message
                _render_feedback(msg, len(st.session_state.messages) - 1)
            except requests.HTTPError as e:
                if e.response.status_code == 401:
                    st.error("Session expired. Please log in again.")
                    st.session_state.clear()
                    st.rerun()
                elif e.response.status_code == 429:
                    st.warning("Too many requests — please wait a moment.")
                else:
                    st.error(f"API error: {e}")


def main() -> None:
    if not st.session_state.get("logged_in"):
        login_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()
