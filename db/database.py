"""SQLite setup, agent action logging and memory reads/writes.

Every read is scoped by client_id. That is a required parameter on every query
function here, not an optional filter, so a caller cannot forget it.
"""

import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from typing import Optional

from data.seed_memory import SEED_MEMORY

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.getenv("RECALL_DB_PATH", os.path.join(PROJECT_ROOT, "recall.db"))


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS client_memory (
            decision_id TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            session_id TEXT,
            decision_type TEXT NOT NULL,
            query_summary TEXT,
            direction TEXT NOT NULL,
            rejection_reason TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS agent_action_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL,
            client_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            action_type TEXT NOT NULL,
            input_summary TEXT,
            retrieval_sources TEXT,
            retrieval_confidence REAL,
            output_summary TEXT,
            human_decision TEXT,
            rejection_reason TEXT,
            analyst_review_required INTEGER DEFAULT 0,
            latency_ms INTEGER,
            model_used TEXT,
            token_count INTEGER,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pending_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id TEXT NOT NULL,
            query TEXT NOT NULL,
            source TEXT DEFAULT 'slack',
            created_at TEXT NOT NULL,
            consumed INTEGER DEFAULT 0
        );
        """
    )
    conn.commit()

    for record in SEED_MEMORY:
        conn.execute(
            """
            INSERT OR IGNORE INTO client_memory
            (decision_id, client_id, decision_type, query_summary, direction,
             rejection_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["decision_id"],
                record["client_id"],
                record["decision_type"],
                record["query_summary"],
                record["direction"],
                record["rejection_reason"],
                record["created_at"],
            ),
        )
    conn.commit()


def log_action(conn, trace_id, client_id, session_id, agent_name, action_type,
               input_summary, retrieval_sources, retrieval_confidence,
               output_summary, human_decision=None, rejection_reason=None,
               analyst_review_required=False, latency_ms=None,
               model_used=None, token_count=None):
    conn.execute(
        """
        INSERT INTO agent_action_log
        (trace_id, client_id, session_id, agent_name, action_type, input_summary,
         retrieval_sources, retrieval_confidence, output_summary, human_decision,
         rejection_reason, analyst_review_required, latency_ms, model_used,
         token_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (trace_id, client_id, session_id, agent_name, action_type,
         input_summary, str(retrieval_sources), retrieval_confidence,
         output_summary, human_decision, rejection_reason,
         int(analyst_review_required), latency_ms, model_used,
         token_count, datetime.now().isoformat()),
    )
    conn.commit()


def write_memory(conn, client_id, session_id, decision_type, query_summary,
                 direction, rejection_reason=None, created_at=None):
    decision_id = f"MEM-{client_id.upper()[:3]}-{uuid.uuid4().hex[:8].upper()}"
    conn.execute(
        """
        INSERT INTO client_memory
        (decision_id, client_id, session_id, decision_type, query_summary,
         direction, rejection_reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (decision_id, client_id, session_id, decision_type, query_summary,
         direction, rejection_reason,
         created_at or datetime.now().strftime("%Y-%m-%d")),
    )
    conn.commit()
    return decision_id


def defer_signal(conn, client_id, session_id, signal_summary, days=30):
    resurface = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
    return write_memory(
        conn, client_id, session_id, "deferred", signal_summary,
        f"Deferred. Re-surface on {resurface}.",
    )


def dismiss_signal(conn, client_id, session_id, signal_summary):
    return write_memory(
        conn, client_id, session_id, "dismissed", signal_summary,
        "Dismissed by client as not relevant.",
    )


def fetch_memory(conn, client_id):
    cursor = conn.execute(
        "SELECT * FROM client_memory WHERE client_id = ? "
        "ORDER BY created_at DESC, rowid DESC",
        (client_id,),
    )
    return [dict(row) for row in cursor.fetchall()]


def fetch_action_log(conn, client_id, session_id):
    cursor = conn.execute(
        "SELECT * FROM agent_action_log WHERE client_id = ? AND session_id = ? "
        "ORDER BY id ASC",
        (client_id, session_id),
    )
    return [dict(row) for row in cursor.fetchall()]


# --------------------------------------------------------------- pending queries
# consumed: 0 = waiting, 1 = picked up and run, 2 = expired unrun (older than the window)
PENDING_QUERY_MAX_AGE_MINUTES = 10


def enqueue_query(conn, client_id, query, source="slack") -> int:
    """A button click somewhere else (Slack) asking the app to run a query."""
    cursor = conn.execute(
        "INSERT INTO pending_queries (client_id, query, source, created_at, consumed) "
        "VALUES (?, ?, ?, ?, 0)",
        (client_id, query, source, datetime.now().isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def claim_pending_query(conn, client_id) -> Optional[dict]:
    """Return the oldest fresh unconsumed query for this client and mark it consumed.

    The UPDATE is conditional on consumed = 0, so if two browser sessions poll at the
    same moment exactly one wins. Queries older than PENDING_QUERY_MAX_AGE_MINUTES are
    expired rather than run: a click from an hour ago must not start a demo by surprise.
    """
    cutoff = (datetime.now() - timedelta(minutes=PENDING_QUERY_MAX_AGE_MINUTES)).isoformat()
    conn.execute(
        "UPDATE pending_queries SET consumed = 2 "
        "WHERE client_id = ? AND consumed = 0 AND created_at < ?",
        (client_id, cutoff),
    )
    row = conn.execute(
        "SELECT * FROM pending_queries WHERE client_id = ? AND consumed = 0 "
        "ORDER BY id ASC LIMIT 1",
        (client_id,),
    ).fetchone()
    if row is None:
        conn.commit()
        return None
    won = conn.execute(
        "UPDATE pending_queries SET consumed = 1 WHERE id = ? AND client_id = ? AND consumed = 0",
        (row["id"], client_id),
    ).rowcount == 1
    conn.commit()
    return dict(row) if won else None
