import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class AppDB:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                );

                CREATE TABLE IF NOT EXISTS indexed_documents (
                    source TEXT PRIMARY KEY,
                    chunk_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_answer TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    answer_mode TEXT NOT NULL,
                    escalate_to_human INTEGER NOT NULL,
                    escalation_reason TEXT,
                    source_summary TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                );

                CREATE TABLE IF NOT EXISTS operator_tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_message TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    escalation_reason TEXT NOT NULL,
                    source_summary TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(id)
                );
                """
            )

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO chat_sessions (id, created_at) VALUES (?, ?)",
                (session_id, now_iso()),
            )
        return session_id

    def ensure_session(self, session_id: str) -> str:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT id FROM chat_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()

            if row is None:
                conn.execute(
                    "INSERT INTO chat_sessions (id, created_at) VALUES (?, ?)",
                    (session_id, now_iso()),
                )

        return session_id

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, now_iso()),
            )

    def get_recent_messages(self, session_id: str, limit: int = 6) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        items = [{"role": row["role"], "content": row["content"]} for row in rows]
        items.reverse()
        return items

    def get_full_history(self, session_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def replace_indexed_documents(self, docs: list[tuple[str, int]]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM indexed_documents")
            conn.executemany(
                """
                INSERT INTO indexed_documents (source, chunk_count, indexed_at)
                VALUES (?, ?, ?)
                """,
                [(source, chunk_count, now_iso()) for source, chunk_count in docs],
            )

    def log_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_answer: str,
        intent: str,
        risk_level: str,
        answer_mode: str,
        escalate_to_human: bool,
        escalation_reason: str | None,
        sources: list[dict],
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_turns (
                    session_id,
                    user_message,
                    assistant_answer,
                    intent,
                    risk_level,
                    answer_mode,
                    escalate_to_human,
                    escalation_reason,
                    source_summary,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_message,
                    assistant_answer,
                    intent,
                    risk_level,
                    answer_mode,
                    1 if escalate_to_human else 0,
                    escalation_reason,
                    json.dumps(sources, ensure_ascii=False),
                    now_iso(),
                ),
            )

    def create_or_get_open_ticket(
        self,
        session_id: str,
        user_message: str,
        intent: str,
        risk_level: str,
        escalation_reason: str,
        sources: list[dict],
    ) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id
                FROM operator_tickets
                WHERE session_id = ?
                  AND status IN ('new', 'in_progress')
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

            if row is not None:
                return int(row["id"])

            cursor = conn.execute(
                """
                INSERT INTO operator_tickets (
                    session_id,
                    user_message,
                    intent,
                    risk_level,
                    escalation_reason,
                    source_summary,
                    status,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'new', ?)
                """,
                (
                    session_id,
                    user_message,
                    intent,
                    risk_level,
                    escalation_reason,
                    json.dumps(sources, ensure_ascii=False),
                    now_iso(),
                ),
            )

            return int(cursor.lastrowid)

    def list_operator_tickets(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    session_id,
                    user_message,
                    intent,
                    risk_level,
                    escalation_reason,
                    source_summary,
                    status,
                    created_at
                FROM operator_tickets
                ORDER BY id DESC
                """
            ).fetchall()

        result: list[dict] = []

        for row in rows:
            result.append(
                {
                    "id": int(row["id"]),
                    "session_id": row["session_id"],
                    "user_message": row["user_message"],
                    "intent": row["intent"],
                    "risk_level": row["risk_level"],
                    "escalation_reason": row["escalation_reason"],
                    "sources": json.loads(row["source_summary"]),
                    "status": row["status"],
                    "created_at": row["created_at"],
                }
            )

        return result

    def update_operator_ticket_status(self, ticket_id: int, status: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE operator_tickets
                SET status = ?
                WHERE id = ?
                """,
                (status, ticket_id),
            )

    def _get_count(self, conn: sqlite3.Connection, query: str, params: tuple = ()) -> int:
        row = conn.execute(query, params).fetchone()
        if row is None:
            return 0
        return int(row[0] or 0)

    def _group_counts(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        field_name: str,
    ) -> list[dict]:
        rows = conn.execute(
            f"""
            SELECT {field_name} AS name, COUNT(*) AS value
            FROM {table_name}
            GROUP BY {field_name}
            ORDER BY value DESC, name ASC
            """
        ).fetchall()

        return [
            {
                "name": str(row["name"]),
                "value": int(row["value"]),
            }
            for row in rows
        ]

    def get_recent_turns(self, limit: int = 15) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    session_id,
                    user_message,
                    assistant_answer,
                    intent,
                    risk_level,
                    answer_mode,
                    escalate_to_human,
                    escalation_reason,
                    source_summary,
                    created_at
                FROM chat_turns
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        result: list[dict] = []

        for row in rows:
            result.append(
                {
                    "id": int(row["id"]),
                    "session_id": row["session_id"],
                    "user_message": row["user_message"],
                    "assistant_answer": row["assistant_answer"],
                    "intent": row["intent"],
                    "risk_level": row["risk_level"],
                    "answer_mode": row["answer_mode"],
                    "escalate_to_human": bool(row["escalate_to_human"]),
                    "escalation_reason": row["escalation_reason"],
                    "sources": json.loads(row["source_summary"]),
                    "created_at": row["created_at"],
                }
            )

        return result

    def get_analytics(self) -> dict:
        with self.connect() as conn:
            total_turns = self._get_count(
                conn,
                "SELECT COUNT(*) FROM chat_turns",
            )
            total_escalations = self._get_count(
                conn,
                "SELECT COUNT(*) FROM chat_turns WHERE escalate_to_human = 1",
            )
            total_tickets = self._get_count(
                conn,
                "SELECT COUNT(*) FROM operator_tickets",
            )

            answer_modes = self._group_counts(conn, "chat_turns", "answer_mode")
            intents = self._group_counts(conn, "chat_turns", "intent")
            risk_levels = self._group_counts(conn, "chat_turns", "risk_level")
            ticket_statuses = self._group_counts(conn, "operator_tickets", "status")

        return {
            "total_turns": total_turns,
            "total_escalations": total_escalations,
            "total_tickets": total_tickets,
            "answer_modes": answer_modes,
            "intents": intents,
            "risk_levels": risk_levels,
            "ticket_statuses": ticket_statuses,
            "recent_turns": self.get_recent_turns(limit=15),
            "recent_tickets": self.list_operator_tickets()[:15],
        }