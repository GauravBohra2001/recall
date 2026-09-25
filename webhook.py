"""Slack interactivity receiver.

Slack POSTs here when someone clicks a button on the Recall notification. Run it next to
the Streamlit app and expose it with ngrok:

    ./venv/bin/uvicorn webhook:app --port 8000

Design points:
  - The signing secret is checked on EVERY request before anything else is read. A request
    that fails, or arrives when no secret is configured, is rejected and does nothing.
  - Slack requires a response within 3 seconds, so the handler returns 200 immediately
    and does the database work in a background task.
  - Slack retries a request it thinks failed (X-Slack-Retry-Num). Retries are acknowledged
    but not processed, so one click can never create two decisions.
  - Credentials are never logged.
"""

import hashlib
import hmac
import json
import logging
import os
import time
from urllib.parse import parse_qs

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request  # noqa: E402

from data.hazra_signals import HAZRA_SIGNALS  # noqa: E402
from db.database import defer_signal, dismiss_signal, get_connection  # noqa: E402

logger = logging.getLogger("recall.webhook")
logging.basicConfig(level=logging.INFO)

CLIENT_ID = os.getenv("CLIENT_ID", "haleon")
SESSION_ID = "slack-webhook"
MAX_REQUEST_AGE_SECONDS = 300  # Slack's own recommendation: reject anything older than 5 minutes

EXPLORE_QUERY = "What should we do about the muscle health opportunity for GLP-1 customers?"
SIGNAL_SUMMARY = HAZRA_SIGNALS[0]["signal_text"][:200]

# Both the button `value` and its `action_id` are accepted. They are set together in db/slack.py.
ACTION_ALIASES = {
    "explore": "explore", "recall_explore": "explore",
    "hold": "hold", "recall_hold": "hold",
    "dismiss": "dismiss", "recall_dismiss": "dismiss",
}

app = FastAPI(title="Recall Slack webhook")


def signing_secret() -> str:
    """Read at request time so a changed .env or test override is picked up."""
    secret = os.getenv("SLACK_SIGNING_SECRET", "")
    return "" if (not secret or "YOUR_" in secret) else secret


def verify_slack_signature(secret: str, timestamp: str, body: bytes, signature: str) -> bool:
    """Slack's v0 scheme: HMAC-SHA256 over 'v0:{timestamp}:{raw body}'."""
    try:
        if abs(time.time() - int(timestamp)) > MAX_REQUEST_AGE_SECONDS:
            return False  # stale, so possibly a replay
    except (TypeError, ValueError):
        return False
    base = b"v0:" + timestamp.encode() + b":" + body
    expected = "v0=" + hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


def extract_action(payload: dict):
    """The normalised action ('explore' / 'hold' / 'dismiss') from a block_actions payload."""
    if payload.get("type") != "block_actions":
        return None
    for action in payload.get("actions", []):
        for key in (action.get("value"), action.get("action_id")):
            if key in ACTION_ALIASES:
                return ACTION_ALIASES[key]
    return None


def handle_action(action: str):
    """Runs after the 200 has been sent. Failures are logged, never raised: Slack has
    already been answered and there is no one to show a traceback to."""
    conn = get_connection()
    try:
        if action == "explore":
            from db.database import enqueue_query  # added with the pending_queries table
            enqueue_query(conn, CLIENT_ID, EXPLORE_QUERY, source="slack")
        elif action == "hold":
            defer_signal(conn, CLIENT_ID, SESSION_ID, SIGNAL_SUMMARY)
        elif action == "dismiss":
            dismiss_signal(conn, CLIENT_ID, SESSION_ID, SIGNAL_SUMMARY)
        logger.info("handled slack action=%s client_id=%s", action, CLIENT_ID)
    except Exception:  # noqa: BLE001
        logger.exception("failed to handle slack action=%s", action)
    finally:
        conn.close()


@app.get("/health")
def health():
    return {"ok": True, "signing_secret_configured": bool(signing_secret())}


@app.post("/slack/actions")
async def slack_actions(request: Request, background: BackgroundTasks):
    secret = signing_secret()
    if not secret:
        # Fail closed: without a secret nothing can be verified, so nothing is accepted.
        raise HTTPException(status_code=503, detail="SLACK_SIGNING_SECRET is not configured")

    body = await request.body()  # the RAW body: the signature is computed over exactly these bytes
    if not verify_slack_signature(
        secret,
        request.headers.get("X-Slack-Request-Timestamp", ""),
        body,
        request.headers.get("X-Slack-Signature", ""),
    ):
        logger.warning("rejected slack request: bad or stale signature")
        raise HTTPException(status_code=401, detail="invalid signature")

    if request.headers.get("X-Slack-Retry-Num"):
        return {"ok": True, "ignored": "retry"}

    try:
        payload = json.loads(parse_qs(body.decode()).get("payload", [""])[0])
    except (ValueError, IndexError):
        raise HTTPException(status_code=400, detail="unreadable payload")

    action = extract_action(payload)
    if action:
        background.add_task(handle_action, action)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
