"""Real Slack notification sender.

Posts the same card that Tab 4 renders, as Block Kit, with three buttons. Returns
(ok, detail) instead of raising, so a missing token on demo day shows a message in the
UI rather than a stack trace. Credentials are read from the environment only and are
never logged.
"""

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger("recall.slack")

PLACEHOLDER_MARKERS = ("YOUR_", "xoxb-your")


def _configured(value: Optional[str]) -> bool:
    return bool(value) and not any(m in value for m in PLACEHOLDER_MARKERS)


def build_blocks(client_id: str, signal_text: str, urgency_tier: str,
                 signal_id: str = None, momentum: float = None,
                 timing_note: str = None, past_decision: str = None) -> list:
    tier = urgency_tier.replace("_", " ").title()
    changed = signal_text
    if momentum is not None:
        changed += f" Momentum {momentum}"
    if signal_id:
        changed += f" [SOURCE: {signal_id}]"

    blocks = [
        {"type": "header", "text": {
            "type": "plain_text",
            "text": f"CREWASIS Recall | {client_id.title()} | GLP-1 Nutrition"}},
        {"type": "section", "text": {
            "type": "mrkdwn", "text": "*Signal moving fast in your category.*"}},
        {"type": "section", "text": {
            "type": "mrkdwn", "text": f"*WHAT CHANGED:* {changed}"}},
        {"type": "section", "text": {
            "type": "mrkdwn",
            "text": f"*URGENCY TIER:* {tier}." + (f" {timing_note}" if timing_note else "")}},
    ]
    if past_decision:
        blocks.append({"type": "section", "text": {
            "type": "mrkdwn", "text": f"*PAST DECISION CONTEXT:* {past_decision}"}})
    blocks.append({
        "type": "actions",
        "elements": [
            {"type": "button", "text": {"type": "plain_text", "text": "Explore This Now"},
             "style": "primary", "value": "explore", "action_id": "recall_explore"},
            {"type": "button", "text": {"type": "plain_text", "text": "Hold for 30 Days"},
             "value": "hold", "action_id": "recall_hold"},
            {"type": "button", "text": {"type": "plain_text", "text": "Not Relevant"},
             "style": "danger", "value": "dismiss", "action_id": "recall_dismiss"},
        ],
    })
    return blocks


def send_notification(client_id: str, signal_text: str, urgency_tier: str,
                      signal_id: str = None, momentum: float = None,
                      timing_note: str = None,
                      past_decision: str = None) -> Tuple[bool, str]:
    token = os.getenv("SLACK_BOT_TOKEN")
    channel = os.getenv("SLACK_CHANNEL_ID")
    if not _configured(token) or not _configured(channel):
        return False, "SLACK_BOT_TOKEN or SLACK_CHANNEL_ID is not set in .env."

    try:
        from slack_sdk import WebClient
        from slack_sdk.errors import SlackApiError
    except ImportError:
        return False, "slack-sdk is not installed. Run pip install -r requirements.txt."

    blocks = build_blocks(client_id, signal_text, urgency_tier, signal_id, momentum,
                          timing_note, past_decision)
    try:
        WebClient(token=token).chat_postMessage(
            channel=channel, blocks=blocks, text="Signal moving fast in your category.")
        return True, "Slack notification sent."
    except SlackApiError as exc:
        error = exc.response.get("error", "unknown")
        hint = " Invite the bot to the channel with /invite @botname." \
            if error in ("not_in_channel", "channel_not_found") else ""
        logger.warning("Slack API error: %s", error)
        return False, f"Slack API error: {error}.{hint}"
    except Exception as exc:  # noqa: BLE001 - network failure, malformed token, etc.
        logger.warning("Slack send failed: %s", type(exc).__name__)
        return False, f"Slack send failed: {type(exc).__name__}"
