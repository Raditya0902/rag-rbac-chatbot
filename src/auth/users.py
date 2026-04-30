import bcrypt


def _hash(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())


USERS_DB = {
    "tony":    {"password": _hash("password123"), "role": "Engineering"},
    "bruce":   {"password": _hash("securepass"),  "role": "Marketing"},
    "sam":     {"password": _hash("financepass"), "role": "Finance"},
    "peter":   {"password": _hash("pete123"),     "role": "Engineering"},
    "sid":     {"password": _hash("sidpass123"),  "role": "Marketing"},
    "natasha": {"password": _hash("hrpass123"),   "role": "HR"},
    "ceo":     {"password": _hash("ceosecret"),   "role": "CLevel"},
}


def authenticate_user(username: str, password: str) -> dict | None:
    user = USERS_DB.get(username.lower())
    if not user:
        return None
    if not bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return None
    return {"username": username.lower(), "role": user["role"]}
