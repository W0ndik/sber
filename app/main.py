from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import re

from app.config import get_settings
from app.db import AppDB
from app.schemas import (
    AnalyticsCountItem,
    AnalyticsResponse,
    AnalyticsTurnItem,
    ChatRequest,
    ChatResponse,
    KnowledgeBaseDocumentItem,
    KnowledgeBaseDocumentListResponse,
    MessageItem,
    OperatorTicketItem,
    OperatorTicketListResponse,
    OperatorTicketStatusUpdate,
    ReindexResponse,
    SessionHistoryResponse,
    SourceItem,
    UploadDocumentsResponse,
)
from app.services.classification import classify_intent
from app.services.escalation import decide_escalation
from app.services.kb_service import KnowledgeBaseService
from app.services.ollama_service import OllamaService
from app.services.vector_store import VectorStore

settings = get_settings()

db = AppDB(settings.sqlite_path)
db.init()

ollama = OllamaService()
vector_store = VectorStore()
kb = KnowledgeBaseService(db=db, ollama=ollama, vector_store=vector_store)

app = FastAPI(
    title="Bank Support AI Assistant",
    version="0.5.6",
)

app.mount("/static", StaticFiles(directory="static"), name="static")


KNOWN_HEADINGS = (
    "Как восстановить доступ в мобильный банк",
    "Как узнать статус заявки на кредит",
    "Как перевыпустить карту",
    "Подозрение на мошенничество и спорные операции",
    "Закрытие счёта",
    "Закрытие счета",
    "Изменение лимита по карте",
    "Где взять реквизиты счёта",
    "Где взять реквизиты счета",
)

INTENT_KEYWORDS = {
    "access_recovery": (
        "доступ",
        "пароль",
        "войти",
        "вход",
        "мобильный банк",
    ),
    "credit_application_status": (
        "статус заявки",
        "заявка",
        "кредит",
    ),
    "card_reissue": (
        "перевыпуск",
        "перевыпустить",
        "карта",
    ),
    "fraud_or_dispute": (
        "мошеннич",
        "оспор",
        "спорн",
        "несанкционирован",
        "списани",
    ),
    "account_closure": (
        "закрыть сч",
        "закрытие сч",
        "расторгнуть",
    ),
    "limit_change": (
        "лимит",
        "изменить лимит",
        "сменить лимит",
    ),
    "bank_details": (
        "реквизит",
        "счет",
        "счёт",
        "iban",
        "swift",
        "бик",
    ),
}

INTENT_SOURCE_HINTS = {
    "access_recovery": ("access_recovery",),
    "credit_application_status": ("credit_status",),
    "card_reissue": ("card_reissue",),
    "fraud_or_dispute": ("fraud",),
    "account_closure": ("account_closure",),
    "limit_change": ("limit_change",),
    "bank_details": ("bank_details",),
}


def normalize_for_match(text: str) -> str:
    return text.lower().replace("ё", "е")


def source_hint_score(intent: str, hit: dict) -> int:
    hints = INTENT_SOURCE_HINTS.get(intent, ())
    if not hints:
        return 0

    source = normalize_for_match(str(hit.get("source", "")))
    score = 0

    for hint in hints:
        if normalize_for_match(hint) in source:
            score += 1

    return score


def text_keyword_score(intent: str, hit: dict) -> int:
    keywords = INTENT_KEYWORDS.get(intent, ())
    if not keywords:
        return 0

    haystack = normalize_for_match(
        f"{hit.get('source', '')} {hit.get('text', '')}"
    )

    score = 0
    for keyword in keywords:
        if normalize_for_match(keyword) in haystack:
            score += 1

    return score


def get_candidate_sources_for_intent(intent: str) -> list[str]:
    hints = INTENT_SOURCE_HINTS.get(intent, ())
    if not hints:
        return []

    documents = kb.list_documents()
    matched_sources: list[str] = []

    for doc in documents:
        filename = str(doc["filename"])
        filename_norm = normalize_for_match(filename)

        for hint in hints:
            if normalize_for_match(hint) in filename_norm:
                matched_sources.append(filename)
                break

    return matched_sources


def restrict_hits_for_intent(intent: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return []

    candidate_sources = get_candidate_sources_for_intent(intent)

    if candidate_sources:
        source_filtered = [
            hit for hit in hits
            if str(hit.get("source", "")) in candidate_sources
        ]
        if source_filtered:
            return source_filtered

    keyword_filtered = [hit for hit in hits if text_keyword_score(intent, hit) > 0]
    if keyword_filtered:
        return keyword_filtered

    return hits


def prioritize_hits_for_intent(intent: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return []

    restricted_hits = restrict_hits_for_intent(intent, hits)
    enriched: list[tuple[int, int, float, dict]] = []

    for hit in restricted_hits:
        src_score = source_hint_score(intent, hit)
        kw_score = text_keyword_score(intent, hit)
        vec_score = float(hit.get("score", 0.0))
        enriched.append((src_score, kw_score, vec_score, hit))

    enriched.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in enriched]


def select_relevant_hits(hits: list[dict]) -> list[dict]:
    if not hits:
        return []

    top_score = hits[0]["score"]

    selected = [
        hit
        for hit in hits
        if hit["score"] >= settings.min_relevance_score
        and hit["score"] >= top_score - 0.08
    ]

    if not selected:
        return [hits[0]]

    return selected[:2]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_known_heading_prefix(text: str) -> str:
    result = text.strip()

    for heading in KNOWN_HEADINGS:
        pattern = rf"^\s*{re.escape(heading)}(?:[\s:.\-–—]+|(?=[А-ЯA-Z]))"
        updated = re.sub(pattern, "", result, flags=re.IGNORECASE).strip()
        if updated != result:
            result = updated

    return result.strip()


def remove_heading(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""

    if len(lines) >= 2:
        first_line = lines[0]
        if any(first_line.lower() == heading.lower() for heading in KNOWN_HEADINGS):
            return normalize_text("\n".join(lines[1:]))

    single = " ".join(lines).strip()
    single = strip_known_heading_prefix(single)
    return normalize_text(single)


def split_sentences(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def build_extractive_answer(chunk_text: str) -> str:
    body = remove_heading(chunk_text)
    body = strip_known_heading_prefix(body)
    sentences = split_sentences(body)

    if not sentences:
        return "Точного подтверждения в базе знаний не найдено."

    result: list[str] = []

    for sentence in sentences:
        sentence = strip_known_heading_prefix(sentence).strip()
        sentence_lower = sentence.lower()

        if not sentence:
            continue

        if "оператор" in sentence_lower:
            continue

        if "ии-помощник" in sentence_lower:
            continue

        result.append(sentence)

        if len(result) >= 2:
            break

    if not result:
        cleaned_body = strip_known_heading_prefix(body).strip()
        if cleaned_body:
            return cleaned_body
        return "Точного подтверждения в базе знаний не найдено."

    answer = " ".join(result).strip()
    answer = strip_known_heading_prefix(answer).strip()
    return answer or "Точного подтверждения в базе знаний не найдено."


def should_use_extractive_answer(
    selected_hits: list[dict],
    escalate: bool,
    intent: str,
) -> bool:
    if escalate:
        return False

    if not selected_hits:
        return False

    if intent in {
        "fraud_or_dispute",
        "account_closure",
        "limit_change",
        "unknown",
    }:
        return False

    if intent == "bank_details":
        return selected_hits[0]["score"] >= 0.20

    if intent in {"access_recovery", "card_reissue"}:
        return selected_hits[0]["score"] >= 0.45

    if len(selected_hits) != 1:
        return False

    if selected_hits[0]["score"] < 0.60:
        return False

    return True


def apply_policy_escalation(
    intent: str,
    risk_level: str,
    escalate: bool,
    reason: str | None,
) -> tuple[bool, str | None]:
    if intent in {"fraud_or_dispute", "account_closure", "limit_change"}:
        return True, reason or "Политика маршрутизации: чувствительный банковский сценарий"

    if intent == "unknown":
        return True, reason or "Политика маршрутизации: запрос не покрыт базой знаний"

    if risk_level in {"high", "critical"}:
        return True, reason or "Политика маршрутизации: повышенный уровень риска"

    return escalate, reason


def get_policy_template_answer(intent: str) -> str | None:
    templates = {
        "fraud_or_dispute": "По такому обращению требуется перевод на оператора.",
        "account_closure": "По вопросу закрытия счёта требуется перевод на оператора.",
        "limit_change": "По вопросу изменения лимита по карте требуется перевод на оператора.",
        "unknown": "Точного подтверждения в базе знаний нет. Требуется перевод на оператора.",
    }
    return templates.get(intent)


def build_messages(
    user_message: str,
    context_blocks: list[str],
    history: list[dict],
    escalate_hint: bool,
) -> list[dict]:
    system_prompt = (
        "Ты ИИ-помощник первой линии поддержки банка. "
        "Отвечай только по предоставленным фрагментам базы знаний и истории диалога. "
        "Используй только ту информацию, которая прямо есть в переданных фрагментах. "
        "Не добавляй факты, которых нет во фрагментах. "
        "Не объединяй разные темы, если они относятся к разным ситуациям. "
        "Не переименовывай роли, подразделения и участников процесса. "
        "Если во фрагменте сказано 'оператор', пиши именно 'оператор'. "
        "Не выдумывай комиссии, сроки, статусы операций, правила банка, юридические нормы, проценты, лимиты и номера документов. "
        "Если в базе знаний нет подтверждения ответа, прямо скажи, что точного подтверждения нет. "
        "Если перевод на оператора не нужен, не упоминай оператора вообще. "
        "Не добавляй общие советы, которых нет во фрагментах. "
        "Пиши по-русски, кратко и по делу."
    )

    context_text = "\n\n".join(
        f"[ФРАГМЕНТ {i + 1}]\n{block}" for i, block in enumerate(context_blocks)
    ).strip()

    if not context_text:
        context_text = "Релевантные фрагменты не найдены."

    if escalate_hint:
        operator_instruction = (
            "Перевод на оператора нужен. "
            "В ответе явно скажи, что нужен оператор."
        )
    else:
        operator_instruction = (
            "Перевод на оператора не нужен. "
            "Не упоминай оператора вообще."
        )

    user_prompt = (
        f"Текущий вопрос клиента:\n{user_message}\n\n"
        f"Фрагменты базы знаний:\n{context_text}\n\n"
        f"Правило по оператору:\n{operator_instruction}\n\n"
        "Дай короткий точный ответ только по этим фрагментам. "
        "Если данных недостаточно, прямо скажи об этом. "
        "Не добавляй информацию из других тем."
    )

    messages = [{"role": "system", "content": system_prompt}]

    for item in history:
        if item["role"] in {"user", "assistant"}:
            messages.append(
                {
                    "role": item["role"],
                    "content": item["content"],
                }
            )

    messages.append({"role": "user", "content": user_prompt})
    return messages


def model_forced_escalation(answer: str) -> bool:
    answer_lower = answer.lower()

    escalation_markers = (
        "нужен оператор",
        "нужно обратиться к оператору",
        "обратитесь к оператору",
        "переведу на оператора",
        "перевод на оператора",
        "требуется оператор",
        "рекомендую обратиться к оператору",
        "рекомендуется обратиться к оператору",
        "нужна помощь оператора",
    )

    return any(marker in answer_lower for marker in escalation_markers)


@app.get("/")
def root() -> FileResponse:
    return FileResponse("static/index.html")


@app.get("/admin")
def admin_page() -> FileResponse:
    return FileResponse("static/admin.html")


@app.get("/operator")
def operator_page() -> FileResponse:
    return FileResponse("static/operator.html")


@app.get("/analytics")
def analytics_page() -> FileResponse:
    return FileResponse("static/analytics.html")


@app.get("/health")
def health() -> dict:
    try:
        tags = ollama.tags()
        return {
            "status": "ok",
            "chat_model": settings.ollama_chat_model,
            "embed_model": settings.ollama_embed_model,
            "ollama_models": [item["name"] for item in tags.get("models", [])],
        }
    except Exception as exc:
        return {
            "status": "degraded",
            "chat_model": settings.ollama_chat_model,
            "embed_model": settings.ollama_embed_model,
            "error": str(exc),
        }


@app.get("/analytics/data", response_model=AnalyticsResponse)
def analytics_data() -> AnalyticsResponse:
    raw = db.get_analytics()

    return AnalyticsResponse(
        total_turns=raw["total_turns"],
        total_escalations=raw["total_escalations"],
        total_tickets=raw["total_tickets"],
        answer_modes=[AnalyticsCountItem(**item) for item in raw["answer_modes"]],
        intents=[AnalyticsCountItem(**item) for item in raw["intents"]],
        risk_levels=[AnalyticsCountItem(**item) for item in raw["risk_levels"]],
        ticket_statuses=[AnalyticsCountItem(**item) for item in raw["ticket_statuses"]],
        recent_turns=[
            AnalyticsTurnItem(
                id=item["id"],
                session_id=item["session_id"],
                user_message=item["user_message"],
                assistant_answer=item["assistant_answer"],
                intent=item["intent"],
                risk_level=item["risk_level"],
                answer_mode=item["answer_mode"],
                escalate_to_human=item["escalate_to_human"],
                escalation_reason=item["escalation_reason"],
                created_at=item["created_at"],
                sources=[
                    SourceItem(
                        source=source["source"],
                        chunk_index=source["chunk_index"],
                        score=round(source["score"], 4),
                    )
                    for source in item["sources"]
                ],
            )
            for item in raw["recent_turns"]
        ],
        recent_tickets=[
            OperatorTicketItem(
                id=item["id"],
                session_id=item["session_id"],
                user_message=item["user_message"],
                intent=item["intent"],
                risk_level=item["risk_level"],
                escalation_reason=item["escalation_reason"],
                status=item["status"],
                created_at=item["created_at"],
                sources=[
                    SourceItem(
                        source=source["source"],
                        chunk_index=source["chunk_index"],
                        score=round(source["score"], 4),
                    )
                    for source in item["sources"]
                ],
            )
            for item in raw["recent_tickets"]
        ],
    )


@app.get("/admin/documents", response_model=KnowledgeBaseDocumentListResponse)
def list_documents() -> KnowledgeBaseDocumentListResponse:
    rows = kb.list_documents()
    return KnowledgeBaseDocumentListResponse(
        documents=[KnowledgeBaseDocumentItem(**row) for row in rows]
    )


@app.post("/admin/upload", response_model=UploadDocumentsResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
) -> UploadDocumentsResponse:
    prepared_files: list[tuple[str, bytes]] = []

    for file in files:
        content = await file.read()
        prepared_files.append((file.filename or "", content))

    saved_files, skipped_files = kb.save_documents(prepared_files)

    return UploadDocumentsResponse(
        saved_files=saved_files,
        skipped_files=skipped_files,
    )


@app.delete("/admin/documents/{filename}")
def delete_document(filename: str) -> dict:
    ok = kb.delete_document(filename)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "ok": True,
        "filename": filename,
    }


@app.post("/admin/reindex", response_model=ReindexResponse)
def reindex_knowledge_base() -> ReindexResponse:
    try:
        files_indexed, chunks_indexed = kb.reindex()
        return ReindexResponse(
            files_indexed=files_indexed,
            chunks_indexed=chunks_indexed,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        if request.session_id:
            session_id = db.ensure_session(request.session_id)
        else:
            session_id = db.create_session()

        history = db.get_recent_messages(session_id, limit=6)

        raw_hits = kb.search(request.message, limit=max(settings.retrieval_limit, 20))
        intent, risk_level = classify_intent(request.message, raw_hits)

        prioritized_hits = prioritize_hits_for_intent(intent, raw_hits)
        selected_hits = select_relevant_hits(prioritized_hits)

        escalate, reason = decide_escalation(
            user_message=request.message,
            hits=selected_hits,
            min_relevance_score=settings.min_relevance_score,
        )

        escalate, reason = apply_policy_escalation(
            intent=intent,
            risk_level=risk_level,
            escalate=escalate,
            reason=reason,
        )

        policy_answer = get_policy_template_answer(intent)

        if policy_answer is not None:
            answer = policy_answer
            answer_mode = "policy_template"
        elif should_use_extractive_answer(selected_hits, escalate, intent):
            answer = build_extractive_answer(selected_hits[0]["text"])
            answer_mode = "extractive"
        else:
            messages = build_messages(
                user_message=request.message,
                context_blocks=[item["text"] for item in selected_hits],
                history=history,
                escalate_hint=escalate,
            )

            answer = ollama.chat(messages=messages, temperature=0.0)
            answer_mode = "llm"

            if (not escalate) and model_forced_escalation(answer):
                escalate = True
                reason = "Модель рекомендовала перевод на оператора"

        operator_ticket_id = None

        if escalate:
            operator_ticket_id = db.create_or_get_open_ticket(
                session_id=session_id,
                user_message=request.message,
                intent=intent,
                risk_level=risk_level,
                escalation_reason=reason or "Требуется проверка оператором",
                sources=selected_hits,
            )

        db.add_message(session_id, "user", request.message)
        db.add_message(session_id, "assistant", answer)

        db.log_turn(
            session_id=session_id,
            user_message=request.message,
            assistant_answer=answer,
            intent=intent,
            risk_level=risk_level,
            answer_mode=answer_mode,
            escalate_to_human=escalate,
            escalation_reason=reason,
            sources=selected_hits,
        )

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            intent=intent,
            risk_level=risk_level,
            answer_mode=answer_mode,
            escalate_to_human=escalate,
            escalation_reason=reason,
            operator_ticket_id=operator_ticket_id,
            sources=[
                SourceItem(
                    source=item["source"],
                    chunk_index=item["chunk_index"],
                    score=round(item["score"], 4),
                )
                for item in selected_hits
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc


@app.get("/sessions/{session_id}", response_model=SessionHistoryResponse)
def get_session_history(session_id: str) -> SessionHistoryResponse:
    messages = db.get_full_history(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        messages=[MessageItem(**item) for item in messages],
    )


@app.get("/operator/tickets", response_model=OperatorTicketListResponse)
def list_operator_tickets() -> OperatorTicketListResponse:
    rows = db.list_operator_tickets()
    return OperatorTicketListResponse(
        tickets=[
            OperatorTicketItem(
                id=row["id"],
                session_id=row["session_id"],
                user_message=row["user_message"],
                intent=row["intent"],
                risk_level=row["risk_level"],
                escalation_reason=row["escalation_reason"],
                status=row["status"],
                created_at=row["created_at"],
                sources=[
                    SourceItem(
                        source=item["source"],
                        chunk_index=item["chunk_index"],
                        score=round(item["score"], 4),
                    )
                    for item in row["sources"]
                ],
            )
            for row in rows
        ]
    )


@app.patch("/operator/tickets/{ticket_id}")
def update_operator_ticket_status(
    ticket_id: int,
    payload: OperatorTicketStatusUpdate,
) -> dict:
    db.update_operator_ticket_status(ticket_id=ticket_id, status=payload.status)
    return {
        "ok": True,
        "ticket_id": ticket_id,
        "status": payload.status,
    }