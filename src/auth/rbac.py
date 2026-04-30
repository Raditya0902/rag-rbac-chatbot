ROLE_NAMESPACE_MAP = {
    "Engineering": ["engineering", "general"],
    "Finance":     ["finance", "general"],
    "HR":          ["hr", "general"],
    "Marketing":   ["marketing", "general"],
    "General":     ["general"],
    "CLevel":      ["engineering", "finance", "hr", "marketing", "general"],
}


def get_allowed_namespaces(role: str) -> list[str]:
    return ROLE_NAMESPACE_MAP.get(role, ["general"])
