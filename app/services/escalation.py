HIGH_RISK_KEYWORDS = (
    "мошеннич",
    "украли карту",
    "похитили карту",
    "заблокировали карту",
    "заблокировали счет",
    "арест счета",
    "списали деньги",
    "несанкционирован",
    "оспорить операц",
    "чужой перевод",
    "утечка данных",
    "смена паспорта",
    "закрыть счет",
    "сменить лимит",
    "подозрительная операция",
)


def decide_escalation(
    user_message: str,
    hits: list[dict],
    min_relevance_score: float,
) -> tuple[bool, str | None]:
    text = user_message.lower()

    if any(keyword in text for keyword in HIGH_RISK_KEYWORDS):
        return True, "Чувствительный банковский сценарий"

    if not hits:
        return True, "В базе знаний не найден релевантный контекст"

    if hits[0]["score"] < min_relevance_score:
        return True, (
            f"Низкая релевантность найденного контекста: "
            f"{hits[0]['score']:.3f} < {min_relevance_score:.3f}"
        )

    return False, None