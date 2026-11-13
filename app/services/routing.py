"""Category -> support team. Deliberately simple; routing accuracy is measured against it."""

TEAM_BY_CATEGORY = {
    "billing": "Billing & Payments",
    "delivery": "Logistics Support",
    "technical": "Technical Support",
    "returns": "Returns & Refunds",
    "account": "Account Security",
}
DEFAULT_TEAM = "General Support"


def route(category: str | None) -> str:
    return TEAM_BY_CATEGORY.get(category or "", DEFAULT_TEAM)
