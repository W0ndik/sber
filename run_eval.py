import csv
import json
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.getenv("BANK_AI_BASE_URL", "http://127.0.0.1:8000")


TEST_CASES = [
    {
        "name": "access_recovery_basic",
        "message": "Как восстановить доступ в мобильный банк?",
        "expected_intent": "access_recovery",
        "expected_escalate": False,
        "expected_risk_level": "medium",
        "expected_answer_mode": "extractive",
        "required_substrings": [
            "забыл пароль",
            "восстановления доступа",
            "подтверждения личности",
        ],
        "forbidden_substrings": [
            "требуется перевод на оператора",
            "нужен оператор",
            "обратитесь к оператору",
        ],
    },
    {
        "name": "access_recovery_cant_login",
        "message": "Я забыл пароль и не могу войти в мобильный банк",
        "expected_intent": "access_recovery",
        "expected_escalate": False,
        "expected_risk_level": "medium",
        "expected_answer_mode": "extractive",
        "required_substrings": [
            "забыл пароль",
            "восстановления доступа",
            "подтверждения личности",
        ],
        "forbidden_substrings": [
            "требуется перевод на оператора",
            "нужен оператор",
            "обратитесь к оператору",
        ],
    },
    {
        "name": "credit_status_basic",
        "message": "Где посмотреть статус заявки на кредит?",
        "expected_intent": "credit_application_status",
        "expected_escalate": False,
        "expected_risk_level": "low",
        "expected_answer_mode": "extractive",
        "required_substrings": [
            "статус заявки",
            "мобильном банке",
            "интернет-банке",
        ],
        "forbidden_substrings": [
            "одобрена",
            "отказана",
            "процентная ставка",
            "требуется перевод на оператора",
        ],
    },
    {
        "name": "card_reissue_basic",
        "message": "Как перевыпустить карту?",
        "expected_intent": "card_reissue",
        "expected_escalate": False,
        "expected_risk_level": "medium",
        "expected_answer_mode": "extractive",
        "required_substrings": [
            "подать заявку",
            "перевыпуск карты",
        ],
        "forbidden_substrings": [
            "как перевыпустить карту клиент",
            "требуется перевод на оператора",
            "нужен оператор",
        ],
    },
    {
        "name": "fraud_charge",
        "message": "У меня списали деньги без моего ведома",
        "expected_intent": "fraud_or_dispute",
        "expected_escalate": True,
        "expected_risk_level": "critical",
        "expected_answer_mode": "policy_template",
        "expected_answer_exact": "По такому обращению требуется перевод на оператора.",
    },
    {
        "name": "fraud_suspicious_operation",
        "message": "Я подозреваю мошенничество по карте",
        "expected_intent": "fraud_or_dispute",
        "expected_escalate": True,
        "expected_risk_level": "critical",
        "expected_answer_mode": "policy_template",
        "expected_answer_exact": "По такому обращению требуется перевод на оператора.",
    },
    {
        "name": "fraud_dispute_operation",
        "message": "Хочу оспорить операцию по карте",
        "expected_intent": "fraud_or_dispute",
        "expected_escalate": True,
        "expected_risk_level": "critical",
        "expected_answer_mode": "policy_template",
        "expected_answer_exact": "По такому обращению требуется перевод на оператора.",
    },
    {
        "name": "account_closure",
        "message": "Хочу закрыть счёт",
        "expected_intent": "account_closure",
        "expected_escalate": True,
        "expected_risk_level": "high",
        "expected_answer_mode": "policy_template",
        "expected_answer_exact": "По вопросу закрытия счёта требуется перевод на оператора.",
    },
    {
        "name": "limit_change",
        "message": "Хочу изменить лимит по карте",
        "expected_intent": "limit_change",
        "expected_escalate": True,
        "expected_risk_level": "high",
        "expected_answer_mode": "policy_template",
        "expected_answer_exact": "По вопросу изменения лимита по карте требуется перевод на оператора.",
    },
    {
        "name": "bank_details",
        "message": "Где взять реквизиты счёта?",
        "expected_intent": "bank_details",
        "expected_escalate": False,
        "expected_risk_level": "low",
        "expected_answer_mode": "extractive",
        "required_substrings": [
            "реквизиты счёта",
            "мобильном банке",
            "интернет-банке",
        ],
        "forbidden_substrings": [
            "мои счета",
            "специалисту первой линии поддержки",
            "обратитесь к специалисту",
            "swift:",
            "iban:",
            "требуется перевод на оператора",
        ],
    },
    {
        "name": "unknown_rejection_reason",
        "message": "Почему банк мне не доверяет?",
        "expected_intent": "unknown",
        "expected_escalate": True,
        "expected_risk_level": "medium",
        "expected_answer_mode": "policy_template",
        "expected_answer_exact": "Точного подтверждения в базе знаний нет. Требуется перевод на оператора.",
    },
    {
        "name": "unknown_mortgage_approval",
        "message": "Когда вы одобрите мне ипотеку?",
        "expected_intent": "unknown",
        "expected_escalate": True,
        "expected_risk_level": "medium",
        "expected_answer_mode": "policy_template",
        "expected_answer_exact": "Точного подтверждения в базе знаний нет. Требуется перевод на оператора.",
    },
]


def normalize_text(text: str) -> str:
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def http_json(method: str, path: str, payload: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    body = None
    headers = {"Accept": "application/json"}

    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    request = Request(url=url, data=body, headers=headers, method=method)

    try:
        with urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {method} {path}: {raw}") from exc
    except URLError as exc:
        raise RuntimeError(f"Connection error for {method} {path}: {exc}") from exc


def check_server() -> dict:
    return http_json("GET", "/health")


def text_contains_all(answer: str, required_substrings: list[str]) -> tuple[bool, str]:
    normalized_answer = normalize_text(answer)
    missing: list[str] = []

    for item in required_substrings:
        if normalize_text(item) not in normalized_answer:
            missing.append(item)

    if missing:
        return False, "missing: " + "; ".join(missing)

    return True, ""


def text_avoids_all(answer: str, forbidden_substrings: list[str]) -> tuple[bool, str]:
    normalized_answer = normalize_text(answer)
    found: list[str] = []

    for item in forbidden_substrings:
        if normalize_text(item) in normalized_answer:
            found.append(item)

    if found:
        return False, "found: " + "; ".join(found)

    return True, ""


def answer_exact_ok(answer: str, expected_answer_exact: str) -> bool:
    return normalize_text(answer) == normalize_text(expected_answer_exact)


def evaluate_case(case: dict) -> dict:
    response = http_json(
        "POST",
        "/chat",
        {
            "message": case["message"],
        },
    )

    actual_intent = response.get("intent")
    actual_escalate = bool(response.get("escalate_to_human"))
    actual_risk_level = response.get("risk_level")
    answer_mode = response.get("answer_mode")
    operator_ticket_id = response.get("operator_ticket_id")
    answer = response.get("answer", "")
    sources = response.get("sources", [])

    intent_ok = actual_intent == case["expected_intent"]
    escalate_ok = actual_escalate == case["expected_escalate"]
    risk_ok = actual_risk_level == case["expected_risk_level"]

    expected_answer_mode = case.get("expected_answer_mode")
    mode_ok = True if expected_answer_mode is None else (answer_mode == expected_answer_mode)

    if actual_escalate:
        ticket_ok = operator_ticket_id is not None
    else:
        ticket_ok = operator_ticket_id is None

    expected_answer_exact = case.get("expected_answer_exact")
    exact_ok = True
    if expected_answer_exact is not None:
        exact_ok = answer_exact_ok(answer, expected_answer_exact)

    required_substrings = case.get("required_substrings", [])
    required_ok, required_details = text_contains_all(answer, required_substrings)

    forbidden_substrings = case.get("forbidden_substrings", [])
    forbidden_ok, forbidden_details = text_avoids_all(answer, forbidden_substrings)

    passed = all(
        (
            intent_ok,
            escalate_ok,
            risk_ok,
            mode_ok,
            ticket_ok,
            exact_ok,
            required_ok,
            forbidden_ok,
        )
    )

    return {
        "name": case["name"],
        "message": case["message"],
        "expected_intent": case["expected_intent"],
        "actual_intent": actual_intent,
        "intent_ok": intent_ok,
        "expected_escalate": case["expected_escalate"],
        "actual_escalate": actual_escalate,
        "escalate_ok": escalate_ok,
        "expected_risk_level": case["expected_risk_level"],
        "actual_risk_level": actual_risk_level,
        "risk_ok": risk_ok,
        "expected_answer_mode": expected_answer_mode,
        "actual_answer_mode": answer_mode,
        "mode_ok": mode_ok,
        "operator_ticket_id": operator_ticket_id,
        "ticket_ok": ticket_ok,
        "source_count": len(sources),
        "expected_answer_exact": expected_answer_exact,
        "exact_ok": exact_ok,
        "required_ok": required_ok,
        "required_details": required_details,
        "forbidden_ok": forbidden_ok,
        "forbidden_details": forbidden_details,
        "answer": answer,
        "passed": passed,
    }


def print_report(rows: list[dict]) -> None:
    print()
    print("Результаты проверки")
    print("-" * 120)

    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        print(
            f"[{status}] {row['name']}\n"
            f"  message: {row['message']}\n"
            f"  intent: {row['actual_intent']} | expected: {row['expected_intent']} | ok={row['intent_ok']}\n"
            f"  escalate: {row['actual_escalate']} | expected: {row['expected_escalate']} | ok={row['escalate_ok']}\n"
            f"  risk: {row['actual_risk_level']} | expected: {row['expected_risk_level']} | ok={row['risk_ok']}\n"
            f"  mode: {row['actual_answer_mode']} | expected: {row['expected_answer_mode']} | ok={row['mode_ok']}\n"
            f"  ticket: {row['operator_ticket_id']} | ticket_ok={row['ticket_ok']} | sources={row['source_count']}\n"
            f"  exact_ok={row['exact_ok']} | required_ok={row['required_ok']} | forbidden_ok={row['forbidden_ok']}\n"
            f"  required_details: {row['required_details'] or '-'}\n"
            f"  forbidden_details: {row['forbidden_details'] or '-'}\n"
            f"  answer: {row['answer']}\n"
        )

    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    failed = total - passed

    print("-" * 120)
    print(f"Итого: total={total}, passed={passed}, failed={failed}")
    print()


def save_csv(rows: list[dict]) -> Path:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = reports_dir / f"eval_report_{timestamp}.csv"

    fieldnames = [
        "name",
        "message",
        "expected_intent",
        "actual_intent",
        "intent_ok",
        "expected_escalate",
        "actual_escalate",
        "escalate_ok",
        "expected_risk_level",
        "actual_risk_level",
        "risk_ok",
        "expected_answer_mode",
        "actual_answer_mode",
        "mode_ok",
        "operator_ticket_id",
        "ticket_ok",
        "source_count",
        "expected_answer_exact",
        "exact_ok",
        "required_ok",
        "required_details",
        "forbidden_ok",
        "forbidden_details",
        "answer",
        "passed",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    return csv_path


def main() -> int:
    print(f"Проверяю сервер: {BASE_URL}")

    try:
        health = check_server()
    except Exception as exc:
        print(f"Не удалось обратиться к серверу: {exc}")
        return 1

    print(f"health: {json.dumps(health, ensure_ascii=False)}")

    rows: list[dict] = []

    for case in TEST_CASES:
        try:
            row = evaluate_case(case)
        except Exception as exc:
            row = {
                "name": case["name"],
                "message": case["message"],
                "expected_intent": case["expected_intent"],
                "actual_intent": None,
                "intent_ok": False,
                "expected_escalate": case["expected_escalate"],
                "actual_escalate": None,
                "escalate_ok": False,
                "expected_risk_level": case["expected_risk_level"],
                "actual_risk_level": None,
                "risk_ok": False,
                "expected_answer_mode": case.get("expected_answer_mode"),
                "actual_answer_mode": None,
                "mode_ok": False,
                "operator_ticket_id": None,
                "ticket_ok": False,
                "source_count": 0,
                "expected_answer_exact": case.get("expected_answer_exact"),
                "exact_ok": False,
                "required_ok": False,
                "required_details": "",
                "forbidden_ok": False,
                "forbidden_details": "",
                "answer": f"ERROR: {exc}",
                "passed": False,
            }

        rows.append(row)

    print_report(rows)
    csv_path = save_csv(rows)
    print(f"CSV-отчёт сохранён: {csv_path}")

    failed = sum(1 for row in rows if not row["passed"])
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())