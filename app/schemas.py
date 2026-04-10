from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class SourceItem(BaseModel):
    source: str
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: str
    risk_level: str
    answer_mode: str
    escalate_to_human: bool
    escalation_reason: str | None = None
    operator_ticket_id: int | None = None
    sources: list[SourceItem]


class ReindexResponse(BaseModel):
    files_indexed: int
    chunks_indexed: int


class MessageItem(BaseModel):
    role: str
    content: str
    created_at: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[MessageItem]


class OperatorTicketItem(BaseModel):
    id: int
    session_id: str
    user_message: str
    intent: str
    risk_level: str
    escalation_reason: str
    status: str
    created_at: str
    sources: list[SourceItem]


class OperatorTicketListResponse(BaseModel):
    tickets: list[OperatorTicketItem]


class OperatorTicketStatusUpdate(BaseModel):
    status: str = Field(pattern="^(new|in_progress|closed)$")


class KnowledgeBaseDocumentItem(BaseModel):
    filename: str
    size_bytes: int
    modified_at: str


class KnowledgeBaseDocumentListResponse(BaseModel):
    documents: list[KnowledgeBaseDocumentItem]


class UploadDocumentsResponse(BaseModel):
    saved_files: list[str]
    skipped_files: list[str]


class AnalyticsCountItem(BaseModel):
    name: str
    value: int


class AnalyticsTurnItem(BaseModel):
    id: int
    session_id: str
    user_message: str
    assistant_answer: str
    intent: str
    risk_level: str
    answer_mode: str
    escalate_to_human: bool
    escalation_reason: str | None = None
    created_at: str
    sources: list[SourceItem]


class AnalyticsResponse(BaseModel):
    total_turns: int
    total_escalations: int
    total_tickets: int
    answer_modes: list[AnalyticsCountItem]
    intents: list[AnalyticsCountItem]
    risk_levels: list[AnalyticsCountItem]
    ticket_statuses: list[AnalyticsCountItem]
    recent_turns: list[AnalyticsTurnItem]
    recent_tickets: list[OperatorTicketItem]