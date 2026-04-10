from app.config import get_settings

settings = get_settings()


def classify_intent(user_message: str, hits: list[dict]) -> tuple[str, str]:
    text = user_message.lower().replace("ё", "е")

    if any(
        keyword in text
        for keyword in (
            "мошеннич",
            "списали деньги",
            "несанкционирован",
            "оспорить операц",
            "чужой перевод",
            "подозрительная операция",
            "утечка данных",
            "украли карту",
            "похитили карту",
        )
    ):
        return "fraud_or_dispute", "critical"

    if any(
        keyword in text
        for keyword in (
            "восстановить доступ",
            "забыл пароль",
            "не могу войти",
            "не удается войти",
            "не получается войти",
            "сбросить пароль",
            "доступ в мобильный банк",
            "вход в мобильный банк",
        )
    ):
        return "access_recovery", "medium"

    if any(
        keyword in text
        for keyword in (
            "статус заявки",
            "заявка на кредит",
            "кредитная заявка",
            "одобрили кредит",
        )
    ):
        return "credit_application_status", "low"

    if any(
        keyword in text
        for keyword in (
            "перевыпустить карту",
            "перевыпуск карты",
            "заменить карту",
            "новая карта",
        )
    ):
        return "card_reissue", "medium"

    if any(
        keyword in text
        for keyword in (
            "закрыть счет",
            "закрыть счёт",
            "закрытие счета",
            "закрытие счёта",
            "расторгнуть договор",
        )
    ):
        return "account_closure", "high"

    if any(
        keyword in text
        for keyword in (
            "изменить лимит",
            "сменить лимит",
            "поднять лимит",
            "снизить лимит",
            "лимит по карте",
        )
    ):
        return "limit_change", "high"

    if any(
        keyword in text
        for keyword in (
            "реквизиты",
            "банковские реквизиты",
            "номер счета",
            "номер счёта",
            "iban",
            "swift",
            "бик",
        )
    ):
        return "bank_details", "low"

    if any(
        keyword in text
        for keyword in (
            "ипотек",
            "не доверя",
            "не доверяет",
            "одобрите мне",
            "почему банк мне не доверяет",
            "почему мне отказали",
            "отказ по ипотеке",
            "когда одобрите",
        )
    ):
        return "unknown", "medium"

    if not hits:
        return "unknown", "medium"

    if hits[0]["score"] < settings.min_relevance_score:
        return "unknown", "medium"

    return "general_faq", "low"