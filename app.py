"""Recall: CREWASIS Hackathon 2026 prototype.

Four tabs, one LangGraph graph, SQLite memory, three real interrupt() human gates.
Run with: streamlit run app.py
"""

import os
import uuid

from dotenv import load_dotenv

load_dotenv()


def _setup_langsmith() -> bool:
    """LangSmith first, before any LangGraph import. Every graph run is then traced with
    no further instrumentation. Without a real key tracing is switched off, not broken."""
    key = os.getenv("LANGCHAIN_API_KEY", "")
    if key and "YOUR_" not in key:
        project = os.getenv("LANGCHAIN_PROJECT", "recall-crewasis-hackathon")
        os.environ.update({
            "LANGCHAIN_TRACING_V2": "true", "LANGCHAIN_API_KEY": key,
            "LANGCHAIN_PROJECT": project,
            "LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": key,
            "LANGSMITH_PROJECT": project,
        })
        return True
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    return False


TRACING_ON = _setup_langsmith()

import html  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import streamlit as st  # noqa: E402
from langgraph.types import Command  # noqa: E402

from agents.graph import build_graph  # noqa: E402
from agents.llm import structured_mode_status  # noqa: E402
from agents.state import new_state  # noqa: E402
from data.hazra_signals import HAZRA_SIGNALS  # noqa: E402
from db.database import (  # noqa: E402
    claim_pending_query,
    defer_signal,
    dismiss_signal,
    fetch_action_log,
    fetch_memory,
    get_connection,
    init_db,
    log_action,
)
from db.slack import send_notification  # noqa: E402

logger = logging.getLogger("recall.app")

CLIENT_ID = os.getenv("CLIENT_ID", "haleon")
PROJECT = os.getenv("LANGCHAIN_PROJECT", "recall-crewasis-hackathon")

QUERY_1 = "What should we do about the muscle health opportunity for GLP-1 customers?"
QUERY_2 = "Draft a marketing claim for our GLP-1 muscle preservation product."
QUERY_3 = "What is the competitive whitespace in GLP-1 nutrition for Haleon?"

TAB_NAMES = ["Main Interaction", "Memory View", "Trace Log", "Proactive Notification"]

st.set_page_config(page_title="Recall | CREWASIS", layout="wide")

st.markdown(
    """
    <style>
      [data-testid="stStatusWidget"] {visibility: hidden;}  /* fragment polling would blink it every 2s */
      .chip {display:inline-block; padding:2px 10px; border-radius:999px;
             font-size:0.82rem; font-weight:600; color:#fff; white-space:nowrap;}
      .chip-xl {font-size:1.08rem; font-weight:800; padding:6px 18px; letter-spacing:.2px;
                box-shadow:0 1px 3px rgba(0,0,0,.25);}
      .tip {cursor:help; opacity:.7; font-size:.9rem; margin-left:4px;}
      .banner {border:1px solid #e2e8f0; border-left:3px solid; border-radius:8px; background:#f8fafc;
               padding:12px 16px; margin:8px 0 14px 0; font-size:13px; color:#0f172a;}
      .banner-orange {border-left-color:#f59e0b;}
      .banner .h {display:block; font-weight:700;}
      .banner ul {margin:6px 0 0 18px; padding:0;}
      .banner li {margin:3px 0; font-size:.92rem;}
      .flagbox {border-left:4px solid #dc2626; background:rgba(220,38,38,.10);
                padding:10px 14px; border-radius:6px; margin-bottom:6px;}
      .okbox {border-left:4px solid #16a34a; background:rgba(22,163,74,.08);
              padding:10px 14px; border-radius:6px; margin-bottom:6px;}
      table.recall {width:100%; border-collapse:collapse; font-size:0.92rem;}
      table.recall th {text-align:left; padding:8px 10px; border-bottom:1px solid #e2e8f0; font-size:10px;
                       font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:#94a3b8;}
      table.recall td {padding:8px 10px; border-bottom:1px solid rgba(128,128,128,.25);
                       vertical-align:top;}
      /* Slack-style message card: the container itself is the card, buttons sit inside it */
      .st-key-slack_card {border:1px solid rgba(128,128,128,.3); border-left:5px solid #2eb67d;
                          border-radius:8px; padding:18px 22px 14px 22px;
                          background:rgba(128,128,128,.05);}
      .slack-head {display:flex; align-items:center; gap:8px; margin-bottom:10px;}
      .slack-avatar {width:36px; height:36px; border-radius:6px; background:#4a154b; color:#fff;
                     display:flex; align-items:center; justify-content:center; font-weight:800;}
      .slack-app {background:rgba(128,128,128,.25); border-radius:3px; padding:0 5px;
                  font-size:.68rem; font-weight:700; letter-spacing:.4px;}
      .slack-time {opacity:.55; font-size:.78rem;}
      .slack-title {font-size:1.15rem; font-weight:800; margin:4px 0 8px 0;}
      .slack-block {margin:7px 0; line-height:1.55;}
      .st-key-btn_explore button, .st-key-btn_hold button,
      .st-key-btn_dismiss button, .st-key-btn_slack button {font-weight:600 !important;}
      .st-key-btn_explore button {background:#007a5a !important; border:1px solid #007a5a !important;}
      .st-key-btn_explore button:hover {background:#005e46 !important;}
      .st-key-btn_hold button {background:#e3e3e3 !important; border:1px solid #b5b5b5 !important;}
      .st-key-btn_hold button:hover {background:#d2d2d2 !important;}
      .st-key-btn_dismiss button {background:#e01e5a !important; border:1px solid #e01e5a !important;}
      .st-key-btn_dismiss button:hover {background:#b8154a !important;}
      .st-key-btn_slack button {background:#4a154b !important; border:1px solid #4a154b !important;}
      .st-key-btn_explore button p, .st-key-btn_dismiss button p,
      .st-key-btn_slack button p {color:#fff !important;}
      .st-key-btn_hold button p {color:#1d1c1d !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <style>
    /* Hide Streamlit chrome. The sidebar re-open control stays visible. */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stExpandSidebarButton"], [data-testid="collapsedControl"] {visibility: visible !important;}

    /* Global typography: system font, a real type scale (13 body / 15 medium / 20 section) */
    :root {color-scheme: light;}
    .stApp {background: #ffffff; color: #0f172a;
            font-family: -apple-system, "Inter", system-ui, sans-serif;}
    .stMarkdown, [data-testid="stMarkdownContainer"], .stButton button, .stTextArea textarea {
        font-family: -apple-system, "Inter", system-ui, sans-serif;}
    h3 {font-size: 20px !important; font-weight: 600 !important; letter-spacing: -0.01em;}

    .block-container {padding-top: 1.5rem !important; padding-bottom: 0rem !important;
                      max-width: 1100px !important;}

    /* Sidebar */
    [data-testid="stSidebar"] {background-color: #0f172a !important;
                               border-right: 1px solid #1e293b !important;
                               min-width: 200px !important;}
    [data-testid="stSidebar"] * {color: #94a3b8 !important;}
    [data-testid="stSidebar"] .brand-title {color: #f1f5f9 !important;}
    [data-testid="stSidebar"] .brand-sub {color: #475569 !important;}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {display: none !important;}
    [data-testid="stSidebar"] [data-testid="stRadioGroup"] {gap: 2px;}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] {
        font-size: 13px !important; padding: 8px 12px !important; border-radius: 6px !important;
        cursor: pointer !important; width: 100%;}
    /* nav items, not radio buttons: hide the circle, mark the current page with a fill */
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {display: none !important;}
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {background: #1e293b !important;}
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover * {color: #f1f5f9 !important;}
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] {background: #1e293b !important;}
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"] * {
        color: #f1f5f9 !important; font-weight: 600 !important;}

    /* Primary actions are slate: colour is reserved for signal, and Approve must not be red */
    [data-testid="stBaseButton-primary"] {background: #0f172a !important; border: 1px solid #0f172a !important;}
    [data-testid="stBaseButton-primary"]:hover {background: #1e293b !important; border-color: #1e293b !important;}
    [data-testid="stBaseButton-primary"] * {color: #ffffff !important;}
    [data-testid="stBaseButton-primary"]:disabled {background: #e2e8f0 !important; border-color: #e2e8f0 !important;}
    [data-testid="stBaseButton-primary"]:disabled * {color: #94a3b8 !important;}

    /* Brief card */
    .brief-card {background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px;
                 padding: 24px 28px; margin: 16px 0;}
    .brief-executive {font-size: 15px; line-height: 1.7; color: #0f172a; font-weight: 450;
                      margin-bottom: 20px;}
    .brief-section-label {font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
                          text-transform: uppercase; color: #94a3b8; margin: 20px 0 8px 0;
                          padding-top: 16px; border-top: 1px solid #f1f5f9;}
    .brief-card > .brief-section-label:first-child {margin-top: 0; padding-top: 0; border-top: none;}
    .confidence-bar {height: 3px; background: #e2e8f0; border-radius: 2px; margin-top: 4px;
                     overflow: hidden;}
    .confidence-fill {height: 100%; background: #16a34a; border-radius: 2px;
                      transition: width 0.6s ease;}
    .stat-chip {display: inline-flex; flex-direction: column; align-items: center;
                background: white; border: 1px solid #e2e8f0; border-radius: 8px;
                padding: 12px 20px; margin-right: 12px;}
    .stat-number {font-size: 22px; font-weight: 600; color: #0f172a; line-height: 1;}
    .stat-label {font-size: 11px; color: #94a3b8; margin-top: 4px; text-transform: uppercase;
                 letter-spacing: 0.05em;}
    .memory-timeline {border-left: 2px solid #f59e0b; padding-left: 16px; margin: 12px 0;}
    .memory-row {display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px;
                 font-size: 13px;}
    .memory-dot-green, .memory-dot-red, .memory-dot-blue {
        width: 8px; height: 8px; border-radius: 50%; margin-top: 4px; flex-shrink: 0;}
    .memory-dot-green {background: #16a34a;}
    .memory-dot-red {background: #dc2626;}
    .memory-dot-blue {background: #3b82f6;}
    .date-badge {font-size: 11px; color: #94a3b8; background: #f8fafc; border: 1px solid #e2e8f0;
                 border-radius: 4px; padding: 1px 6px; white-space: nowrap;}
    .agent-label {font-size: 10px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
                  color: #94a3b8; margin-bottom: 6px;}
    .agent-state-panel {background: #0f172a; border-radius: 8px; padding: 14px;
                        font-family: "SF Mono", "Fira Code", monospace; font-size: 12px; color: #94a3b8;}
    .state-key {color: #7dd3fc;}
    .state-string {color: #86efac;}
    .state-null {color: #64748b;}
    .state-bool {color: #f59e0b;}
    .step-item {padding: 6px 0; border-bottom: 1px solid #1e293b; font-size: 12px;}
    .step-item:last-child {border-bottom: none;}
    .step-item .n {color: #475569; display: inline-block; width: 1.6rem;}
    .step-item .d {display: block; margin: 2px 0 0 1.6rem; font-size: 11px; color: #64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------ resources
@st.cache_resource
def get_db():
    conn = get_connection()
    init_db(conn)
    return conn


@st.cache_resource
def get_graph(_conn):
    return build_graph(_conn)


@st.cache_resource(show_spinner="Loading embedding model...")
def warm_models():
    from rag.retrieval import embed
    embed([s["signal_text"] for s in HAZRA_SIGNALS])
    return True


conn = get_db()
graph = get_graph(conn)
warm_models()

defaults = {
    "session_id": uuid.uuid4().hex[:12],
    "nav": TAB_NAMES[0],
    "active_tab": 0,
    "langgraph_state": {"status": "idle", "client_id": CLIENT_ID},
    "state_history": [],
    "run_config": None,
    "run_id": None,
    "query_text": QUERY_1,
    "autorun": False,
    "pipeline_error": None,
    "show_reject": False,
    "rejected_claims": {},
    "flash": None,
    "trace_url": None,
}
for _k, _v in defaults.items():
    st.session_state.setdefault(_k, _v)


# ---------------------------------------------------------------------------- html
def flat(markup: str) -> str:
    """One line of HTML. st.markdown treats a blank line plus an indented line as a code
    block, so multi-line HTML templates are collapsed before they are rendered."""
    return " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())


# Lucide icons (lucide.dev, ISC licence), the inner markup of each 24x24 stroke icon.
LUCIDE = {
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
             '<path d="M12 9v4"/><path d="M12 17h.01"/>',
    "history": '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>'
               '<path d="M12 7v5l4 2"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
}


def icon(name: str, size: int = 14, color: str = "currentColor") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align:-2px;flex-shrink:0">{LUCIDE[name]}</svg>')


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def chip(label: str, color: str, title: str = None, big: bool = False) -> str:
    cls = "chip chip-xl" if big else "chip"
    tip = f' title="{esc(title)}"' if title else ""
    return f'<span class="{cls}" style="background:{color}"{tip}>{esc(label)}</span>'


PILL_STYLES = {
    "src": ("#dbeafe", "#1e40af", "#93c5fd"),   # citation badge: light blue pill
    "mem": ("#ede9fe", "#5b21b6", "#c4b5fd"),   # memory reference: light violet pill
}


def pill(text: str, kind: str = "src") -> str:
    bg, fg, border = PILL_STYLES[kind]
    return (f'<span style="display:inline-block;background:{bg};color:{fg};'
            f'border:1px solid {border};border-radius:999px;padding:0 9px;font-size:0.78rem;'
            f'font-weight:600;line-height:1.65;white-space:nowrap;vertical-align:baseline;'
            f'margin:0 1px;">{esc(text)}</span>')


def highlight_sources(text: str) -> str:
    """Escape first, then turn [SOURCE: id] tags into light-blue citation pills."""
    parts = re.split(r"(\[SOURCE: [^\]]+\])", str(text if text is not None else ""))
    out = []
    for part in parts:
        m = re.fullmatch(r"\[SOURCE: ([^\]]+)\]", part)
        out.append(pill("SOURCE: " + m.group(1).strip()) if m else esc(part))
    return "".join(out)


CONFIDENCE_TIP = ("Weighted average of HAZRA momentum \u00d7 confidence scores for "
                  "retrieved signals.")


def render_confidence_badge(score: float, analyst_review: bool = False):
    color = "#16a34a" if score >= 0.75 else "#d97706" if score >= 0.50 else "#dc2626"
    label = ("High confidence" if score >= 0.75
             else "Analyst review required" if score >= 0.50 else "Insufficient signal")
    if analyst_review and score >= 0.50:
        label = "Analyst review required"
    fill_pct = int(score * 100)
    st.markdown(flat(f"""
    <div title="{esc(CONFIDENCE_TIP)}" style="display:inline-flex; flex-direction:column;
         gap:4px; margin:12px 0; cursor:help;">
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:22px; font-weight:700; color:{color}; line-height:1;">{score:.2f}</span>
            <div>
                <div style="font-size:11px; color:{color}; font-weight:500;">{esc(label)}</div>
                <div style="font-size:10px; color:#94a3b8; display:flex; align-items:center; gap:4px;">
                    confidence score {icon("info", 11, "#94a3b8")}
                </div>
            </div>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{fill_pct}%; background:{color};"></div>
        </div>
    </div>
    """), unsafe_allow_html=True)


def render_banner(kind: str, header: str, items=()):
    """kind: yellow = precedent found (reinforcing), orange = conflict (needs a decision)."""
    lis = "".join(f"<li>{highlight_sources(i)}</li>" for i in items)
    body = f"<ul>{lis}</ul>" if lis else ""
    st.markdown(f'<div class="banner banner-{kind}"><span class="h">{esc(header)}</span>'
                f"{body}</div>", unsafe_allow_html=True)


def render_memory_banner(memory_context: list, memory_finding: str):
    if not memory_context or memory_finding == "no_history":
        return

    conflict = memory_finding == "conflict_found"
    color = "#f59e0b" if conflict else "#3b82f6"
    title = "Memory conflict" if conflict else "Memory context"
    subtitle = ("This query resembles a direction previously rejected." if conflict else
                "Past decision found. Memory context is incorporated in this brief.")
    bg = "#fffbeb" if conflict else "#eff6ff"
    border = "#fbbf24" if conflict else "#93c5fd"

    rows_html = ""
    for m in memory_context:
        dt = m.get("decision_type", "")
        if "approved" in dt or "selected" in dt:
            dot_class = "memory-dot-green"
        elif "rejected" in dt:
            dot_class = "memory-dot-red"
        else:
            dot_class = "memory-dot-blue"   # deferred / dismissed: neither a yes nor a no
        date = (m.get("created_at") or "")[:10]
        direction = re.sub(r"\s*\[SOURCE: [^\]]+\]", "", m.get("direction") or "").strip()
        direction = direction[:80] + ("..." if len(direction) > 80 else "")
        reason = m.get("rejection_reason") or ""
        reason_html = (f'<div style="font-size:11px; color:#dc2626; margin-top:2px;">'
                       f'{esc(reason[:60])}{"..." if len(reason) > 60 else ""}</div>') if reason else ""
        rows_html += flat(f"""
        <div class="memory-row">
            <div class="{dot_class}"></div>
            <div>
                <span class="date-badge">{esc(date)}</span>
                <span style="font-size:11px; color:#64748b; margin-left:6px;">{esc(dt.replace("_", " "))}</span>
                <div style="font-size:13px; color:#1e293b; margin-top:2px;">{esc(direction)}</div>
                {reason_html}
            </div>
        </div>""")

    st.markdown(flat(f"""
    <div style="background:{bg}; border:1px solid {border}; border-left:3px solid {color};
         border-radius:8px; padding:14px 16px; margin:12px 0;">
        <div style="font-size:13px; font-weight:600; color:{color}; margin-bottom:4px;
             display:flex; align-items:center; gap:6px;">
            {icon("alert" if conflict else "history", 15, color)}{title}</div>
        <div style="font-size:12px; color:#64748b; margin-bottom:12px;">{subtitle}</div>
        <div class="memory-timeline" style="border-left-color:{color};">{rows_html}</div>
    </div>
    """), unsafe_allow_html=True)


URGENCY = {
    "act_this_quarter": ("Act This Quarter", "#dc2626"),
    "act_this_year": ("Act This Year", "#ea580c"),
    "monitor": ("Monitor", "#6b7280"),
}

DECISION_COLORS = {
    "brief_approved": "#16a34a", "claim_selected": "#2563eb",
    "claim_rejected": "#dc2626", "brief_rejected": "#dc2626",
    "deferred": "#ea580c", "dismissed": "#6b7280",
}


# ---------------------------------------------------------------------- graph runner
class StatePanel:
    """The right-hand column. Redrawn on every agent step, so the JSON and the step
    history are live during a run instead of lagging one rerun behind."""

    def __init__(self, json_slot, history_slot):
        self.json_slot, self.history_slot = json_slot, history_slot

    def json(self, ui: dict):
        self.json_slot.json(ui)
        self.draw_history()

    def draw_history(self):
        steps = st.session_state["state_history"]
        rows = []
        for i, step in enumerate(steps, 1):
            status = step.get("status", "")
            looping = status == "secondary_retrieval" or (
                status == "sufficiency_check" and step.get("secondary_retrieval_needed"))
            detail = " \u00b7 ".join(f"{k}: {v}" for k, v in step.items() if k != "status")
            mark = "\u21bb " if status == "secondary_retrieval" else ""
            rows.append(
                f'<div class="step-item"><span class="n">{i}.</span>'
                f'<span class="{"state-bool" if looping else "state-key"}" '
                f'style="font-weight:600">{mark}{esc(status)}</span>'
                + (f'<span class="d">{esc(detail[:170])}</span>' if detail else "")
                + "</div>")
        with self.history_slot.container():
            with st.expander("Step history", expanded=True):
                if rows:
                    st.markdown('<div class="agent-state-panel">' + "".join(rows) + "</div>",
                                unsafe_allow_html=True)
                else:
                    st.caption("Agent steps appear here as the graph runs.")


def push_state(ui: dict, panel):
    st.session_state["langgraph_state"] = ui
    st.session_state["state_history"].append(ui)
    panel.json(ui)


def stream_graph(graph_input, config, panel):
    """Run the graph until it finishes or hits an interrupt(). Each agent step updates
    the state panel live. Nothing here can surface a traceback to the analyst."""
    try:
        for chunk in graph.stream(graph_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__" or not isinstance(update, dict):
                    continue
                for entry in update.get("agent_log", []):
                    if entry.get("ui_state"):
                        push_state(entry["ui_state"], panel)
                        time.sleep(0.25)
    except Exception as exc:  # noqa: BLE001
        logger.exception("pipeline failed")
        cfg = config["configurable"]
        log_action(
            conn, cfg["trace_id"], CLIENT_ID, st.session_state["session_id"],
            "orchestrator", "pipeline_error", None, [], None,
            f"{type(exc).__name__}: {str(exc)[:200]}",
        )
        st.session_state["pipeline_error"] = (
            "The pipeline stopped before finishing. Nothing was written to memory. "
            "Tab 3 shows every agent action that did run. "
            f"({type(exc).__name__})"
        )


def start_run(query: str, panel):
    trace_id = uuid.uuid4().hex[:12]
    run_id = str(uuid.uuid4())
    config = {
        "configurable": {"thread_id": uuid.uuid4().hex, "trace_id": trace_id},
        "run_id": run_id,
        "run_name": "recall_query",
        "tags": ["recall", CLIENT_ID],
        "metadata": {"trace_id": trace_id, "client_id": CLIENT_ID,
                     "session_id": st.session_state["session_id"]},
    }
    st.session_state.update(
        run_config=config, run_id=run_id, state_history=[], pipeline_error=None,
        show_reject=False, rejected_claims={}, trace_url=None,
        langgraph_state={"status": "starting", "client_id": CLIENT_ID},
    )
    state = new_state(CLIENT_ID, st.session_state["session_id"], trace_id, query)
    stream_graph(state, config, panel)


def resume_run(decision: dict, panel):
    """Resume the checkpointed thread with the human's decision, then redraw."""
    stream_graph(Command(resume=decision), st.session_state["run_config"], panel)
    st.session_state["show_reject"] = False
    st.rerun()


def snapshot():
    """Source of truth is the checkpointer: (state values, pending interrupt payload)."""
    config = st.session_state.get("run_config")
    if not config:
        return None, None
    snap = graph.get_state(config)
    payload = None
    if snap.next:
        for task in snap.tasks:
            if task.interrupts:
                payload = task.interrupts[0].value
                break
    return snap.values, payload


# ------------------------------------------------------------------------- renderers
def render_brief_section(brief_data: dict):
    exec_summary = brief_data.get("executive_summary", "")
    key_findings = brief_data.get("key_findings", [])
    recommended = brief_data.get("recommended_direction", "")
    urgency = brief_data.get("urgency_tier", "monitor")
    memory_note = brief_data.get("memory_incorporated", "")

    urgency_color = {"act_this_quarter": "#dc2626", "act_this_year": "#d97706",
                     "monitor": "#64748b"}.get(urgency, "#64748b")
    urgency_label = urgency.replace("_", " ").title()

    findings_html = "".join(
        f'<div style="padding:8px 0; border-bottom:1px solid #f8fafc; font-size:13px; '
        f'color:#334155; line-height:1.6;">{highlight_sources(f)}</div>'
        for f in key_findings)

    memory_html = flat(f"""
    <div class="brief-section-label">Memory incorporated</div>
    <div style="font-size:13px; color:#475569; line-height:1.6; background:#f8fafc;
         border-radius:6px; padding:12px; border-left:3px solid #3b82f6;">{esc(memory_note)}</div>
    """) if memory_note else ""

    st.markdown(flat(f"""
    <div class="brief-card">
        <div class="brief-section-label">Executive summary</div>
        <div class="brief-executive">{highlight_sources(exec_summary)}</div>
        <div class="brief-section-label">Key findings</div>
        {findings_html}
        <div class="brief-section-label">Recommended direction</div>
        <div style="font-size:14px; color:#0f172a; font-weight:500; line-height:1.6;
             padding:8px 0;">{highlight_sources(recommended)}</div>
        <div class="brief-section-label">Urgency</div>
        <div style="display:inline-flex; align-items:center; gap:6px;
             background:{urgency_color}15; border:1px solid {urgency_color}40;
             color:{urgency_color}; font-size:13px; font-weight:600;
             padding:5px 12px; border-radius:20px; margin-top:4px;">
            {icon("clock", 14, urgency_color)}{esc(urgency_label)}</div>
        {memory_html}
    </div>
    """), unsafe_allow_html=True)


def gate_conflict(payload, panel):
    render_banner(
        "orange",
        "Analyst decision required. The strategy agent will not run until you "
        "acknowledge this conflict.",
    )
    c1, c2, _ = st.columns([3, 1.2, 2.5])
    if c1.button("Acknowledge conflict and proceed", type="primary", use_container_width=True):
        resume_run({"decision": "proceed"}, panel)
    if c2.button("Stop here"):
        resume_run({"decision": "cancel"}, panel)


def gate_brief(payload, values, panel):
    brief = json.loads(payload["brief"])
    render_confidence_badge(payload["confidence"], payload.get("analyst_review_required", False))
    render_brief_section(brief)

    st.markdown("**Awaiting human approval.** Nothing is written to memory until you decide.")
    _, col_b, col_c = st.columns([2, 1, 1])
    if col_b.button("Approve", icon=":material/check:", type="primary", use_container_width=True):
        resume_run({"decision": "approved"}, panel)
    if col_c.button("Reject", icon=":material/close:", use_container_width=True):
        st.session_state["show_reject"] = True
        st.rerun()

    if st.session_state["show_reject"]:
        reason = st.text_input("Rejection reason (required, saved to memory)",
                               key="brief_reason")
        if st.button("Confirm rejection"):
            if not reason.strip():
                st.error("A reason is required. Memory is only useful if it records why.")
            else:
                resume_run({"decision": "rejected", "reason": reason.strip()}, panel)


def gate_claims(payload, panel):
    claims = payload["claims"]
    rejected = st.session_state["rejected_claims"]
    render_confidence_badge(payload["confidence"], False)
    st.markdown(
        "**Three claim options.** Options in red match a decision your legal team "
        "already rejected. Reject them, then select a safe option."
    )

    for claim in claims:
        cid = claim["claim_id"]
        strength = esc(claim.get("claim_strength", ""))
        if claim.get("flagged"):
            struck = " (rejected)" if cid in rejected else ""
            st.markdown(
                f'<div class="flagbox"><b style="color:#dc2626">FLAGGED {esc(cid)}'
                f'{struck}</b> ({strength})<br>{esc(claim["claim_text"])}<br>'
                f'<span style="color:#dc2626">{esc(claim.get("flag_reason"))}</span><br>'
                f'<small>{highlight_sources(claim.get("rationale", ""))}</small></div>',
                unsafe_allow_html=True)
            if cid not in rejected and st.button(f"Reject {cid}", key=f"rej_{cid}"):
                st.session_state["rejected_claims"][cid] = claim["flag_reason"]
                st.rerun()
        else:
            st.markdown(
                f'<div class="okbox"><b>{esc(cid)}</b> ({strength})<br>'
                f'{esc(claim["claim_text"])}<br>'
                f'<small>{highlight_sources(claim.get("rationale", ""))}</small></div>',
                unsafe_allow_html=True)

    safe = [c for c in claims if not c.get("flagged") and c["claim_id"] not in rejected]
    st.divider()
    st.markdown("**Awaiting human selection.**")
    choice = st.radio("Select the claim to take forward", [c["claim_id"] for c in safe],
                      format_func=lambda cid: next(
                          c["claim_text"] for c in claims if c["claim_id"] == cid),
                      index=None) if safe else None
    if st.button("Confirm selection", type="primary", disabled=choice is None):
        resume_run({
            "selected_claim_id": choice,
            "rejected": [{"claim_id": k, "reason": v} for k, v in rejected.items()],
        }, panel)


# ---------------------------------------------------------------------------- tab 1
def tab_main():
    left, right = st.columns([7, 3])

    with right:
        st.markdown('<div class="agent-label">Agent state <span style="opacity:.6">'
                    '&middot; LangGraph</span></div>', unsafe_allow_html=True)
        panel = StatePanel(st.empty(), st.empty())
        panel.json_slot.json(st.session_state["langgraph_state"])
        panel.draw_history()

    with left:
        if st.session_state.get("auto_query"):
            st.session_state["_qb"] = st.session_state.pop("auto_query")
            st.session_state["autorun"] = True
        st.session_state.setdefault("_qb", st.session_state["query_text"])

        st.subheader("Client query")
        st.text_area("Query", key="_qb", height=80, label_visibility="collapsed")
        st.session_state["query_text"] = st.session_state["_qb"]

        def _set(q):
            st.session_state["_qb"] = q

        cols = st.columns([1, 1, 1, 1, 3])
        run = cols[0].button("Run", type="primary")
        cols[1].button("Query 1", on_click=_set, args=(QUERY_1,))
        cols[2].button("Query 2", on_click=_set, args=(QUERY_2,))
        cols[3].button("Query 3", on_click=_set, args=(QUERY_3,))

        if st.session_state["autorun"]:
            st.session_state["autorun"] = False
            run = True
        if run and st.session_state["query_text"].strip():
            with st.spinner("Agents running..."):
                start_run(st.session_state["query_text"].strip(), panel)

        values, payload = snapshot()
        if not values or not values.get("query"):
            st.info("Run a query to start. Memory is read first, then signal is retrieved.")
            return

        render_memory_banner(values.get("memory_context") or [], values.get("memory_finding"))

        if st.session_state["pipeline_error"]:
            st.error(st.session_state["pipeline_error"])
        if values.get("error_message"):
            st.error(values["error_message"])
            if values.get("retrieval_attempts"):
                render_confidence_badge(values.get("retrieval_confidence", 0.0), False)

        if payload:
            kind = payload["type"]
            if kind == "conflict_gate":
                gate_conflict(payload, panel)
            elif kind == "brief_approval":
                gate_brief(payload, values, panel)
            elif kind == "claim_selection":
                gate_claims(payload, panel)
            return

        # Graph finished: show the outcome read-only.
        if values.get("draft_brief") and values.get("human_brief_decision"):
            render_confidence_badge(values["brief_confidence"],
                                    values.get("analyst_review_required", False))
            render_brief_section(json.loads(values["draft_brief"]))
            verdict = values["human_brief_decision"]
            if verdict == "approved":
                st.success("Brief approved and written to memory. See Memory View.")
            else:
                st.error(f"Brief rejected: {values.get('brief_rejection_reason')}. "
                         "Written to memory so it is not proposed again.")
        elif values.get("draft_claims") and values.get("memory_written"):
            chosen = next((c for c in values["draft_claims"]
                           if c["claim_id"] == values.get("selected_claim")), None)
            for c in values["draft_claims"]:
                if c.get("rejected"):
                    st.error(f"{c['claim_id']} rejected: {c['claim_text']}")
            if chosen:
                st.success(f"Selected {chosen['claim_id']}: {chosen['claim_text']}")
            st.caption("Decisions written to memory. See Memory View.")


# ---------------------------------------------------------------------------- tab 2
def tab_memory():
    st.subheader(f"Client memory: {CLIENT_ID.title()}")
    records = fetch_memory(conn, CLIENT_ID)
    total = len(records)
    approved = sum(1 for r in records
                   if "approved" in r["decision_type"] or "selected" in r["decision_type"])
    rejected = sum(1 for r in records if "rejected" in r["decision_type"])
    other = total - approved - rejected  # deferred / dismissed are neither
    other_chip = (f'<div class="stat-chip"><div class="stat-number" style="color:#64748b;">{other}</div>'
                  f'<div class="stat-label">Deferred / dismissed</div></div>') if other else ""
    st.markdown(flat(f"""
    <div style="display:flex; gap:12px; margin-bottom:20px;">
        <div class="stat-chip"><div class="stat-number">{total}</div>
            <div class="stat-label">Total decisions</div></div>
        <div class="stat-chip"><div class="stat-number" style="color:#16a34a;">{approved}</div>
            <div class="stat-label">Approved</div></div>
        <div class="stat-chip"><div class="stat-number" style="color:#dc2626;">{rejected}</div>
            <div class="stat-label">Rejected</div></div>
        {other_chip}
    </div>
    """), unsafe_allow_html=True)
    rows = "".join(
        f"<tr><td>{esc(r['created_at'])}</td><td>{esc(r['query_summary'])}</td>"
        f"<td>{chip(r['decision_type'].replace('_', ' '), DECISION_COLORS.get(r['decision_type'], '#6b7280'))}</td>"
        f"<td>{highlight_sources(r['direction'])}</td><td>{esc(r['rejection_reason'])}</td></tr>"
        for r in records
    )
    st.markdown(
        "<table class='recall'><tr><th>Date</th><th>Query Summary</th><th>Decision Type</th>"
        f"<th>Direction</th><th>Rejection Reason</th></tr>{rows}</table>",
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------- tab 3
def fetch_trace_url(run_id: str):
    try:
        from langsmith import Client
        return Client().read_run(run_id).url
    except Exception:  # noqa: BLE001 - traces flush asynchronously; not-yet-there is normal
        return None


def tab_trace():
    st.subheader("Trace log: current session")
    st.caption(f"session_id {st.session_state['session_id']} | client_id {CLIENT_ID}")

    rows = fetch_action_log(conn, CLIENT_ID, st.session_state["session_id"])
    if not rows:
        st.info("No agent actions yet this session. Run a query in Tab 1.")
    else:
        st.dataframe([{
            "Timestamp": r["created_at"][11:19],
            "Agent": r["agent_name"],
            "Action": r["action_type"],
            "Sources": r["retrieval_sources"],
            "Confidence": r["retrieval_confidence"],
            "Output": (r["output_summary"] or "")[:140],
            "Human Decision": r["human_decision"] or "",
            "Latency ms": r["latency_ms"],
        } for r in rows], use_container_width=True, hide_index=True, column_config={
            "Timestamp": st.column_config.TextColumn(width=78),
            "Agent": st.column_config.TextColumn(width=122),
            "Action": st.column_config.TextColumn(width=170),
            "Sources": st.column_config.TextColumn(width=104),
            "Confidence": st.column_config.NumberColumn(width=92),
            "Output": st.column_config.TextColumn(width=160),
            "Human Decision": st.column_config.TextColumn(width=118),
            "Latency ms": st.column_config.NumberColumn(width=92),
        })

    st.markdown("#### LangSmith")
    mode = structured_mode_status()
    if mode["fallback_reason"]:
        st.warning(f"Structured output is running in **{mode['mode']}**. "
                   f"json_schema was rejected by the deployment: {mode['fallback_reason']}")
    else:
        st.caption(f"Structured output mode: {mode['mode']}")

    if not TRACING_ON:
        st.warning("LANGCHAIN_API_KEY is not set, so tracing is off.")
        return
    run_id = st.session_state.get("run_id")
    if not run_id:
        st.caption(f"Project {PROJECT}. Run a query to get a trace link.")
        return
    if st.button("Get LangSmith trace URL for the latest run") or st.session_state["trace_url"]:
        if not st.session_state["trace_url"]:
            st.session_state["trace_url"] = fetch_trace_url(run_id)
        if st.session_state["trace_url"]:
            st.markdown(f"[Open trace in LangSmith]({st.session_state['trace_url']})")
        else:
            st.info(f"Trace not indexed yet. Try again in a few seconds. Run id: `{run_id}`, "
                    f"project: {PROJECT}.")


# ---------------------------------------------------------------------------- tab 4
SIGNAL_1 = HAZRA_SIGNALS[0]
SIGNAL_3 = HAZRA_SIGNALS[2]
TIMING_NOTE = ("Based on historical category response time first competitor launch "
               "expected within 12-16 months. Current signal age 8 months.")
PAST_DECISION = ("You approved muscle preservation direction March 2024. This signal "
                 "reinforces that direction.")


def tab_notification():
    st.subheader("Proactive notification")
    st.caption("What Recall pushes to Slack when HAZRA detects a fast-moving signal.")

    if st.session_state["flash"]:
        st.success(st.session_state["flash"])
        st.session_state["flash"] = None

    sent_at = datetime.now().strftime("%I:%M %p").lstrip("0")
    with st.container(key="slack_card"):
        st.markdown(
            '<div class="slack-head"><span class="slack-avatar">R</span>'
            '<b>CREWASIS Recall</b><span class="slack-app">APP</span>'
            f'<span class="slack-time">{sent_at}</span></div>'
            f'<div class="slack-title">CREWASIS Recall | {esc(CLIENT_ID.title())} | GLP-1 Nutrition</div>'
            '<div class="slack-block"><b>Signal moving fast in your category.</b></div>'
            '<div class="slack-block"><b>WHAT CHANGED:</b> Consumer concern about muscle loss '
            "on GLP-1 medications has accelerated. "
            f"Momentum {SIGNAL_1['hazra_momentum_score']} "
            f"{highlight_sources('[SOURCE: ' + SIGNAL_1['signal_id'] + ']')}</div>"
            f'<div class="slack-block"><b>URGENCY TIER:</b> Act This Year. {esc(TIMING_NOTE)} '
            f"{highlight_sources('[SOURCE: ' + SIGNAL_3['signal_id'] + ']')}</div>"
            f'<div class="slack-block"><b>PAST DECISION CONTEXT:</b> {esc(PAST_DECISION)} '
            f'{pill("MEM-HAL-2024-001", "mem")}</div>',
            unsafe_allow_html=True)

        c1, c2, c3, _ = st.columns([1.3, 1.3, 1.3, 2.5])
        if c1.button("Explore This Now", key="btn_explore", use_container_width=True):
            st.session_state["auto_query"] = (
                "What should we do about the muscle health opportunity for GLP-1 customers?")
            st.session_state["active_tab"] = 0
            st.session_state["switch_tab"] = True
            st.rerun()
        if c2.button("Hold for 30 Days", key="btn_hold", use_container_width=True):
            defer_signal(conn, CLIENT_ID, st.session_state["session_id"],
                         SIGNAL_1["signal_text"][:200])
            st.session_state["flash"] = "Deferred for 30 days and written to memory."
            st.rerun()
        if c3.button("Not Relevant", key="btn_dismiss", use_container_width=True):
            dismiss_signal(conn, CLIENT_ID, st.session_state["session_id"],
                           SIGNAL_1["signal_text"][:200])
            st.session_state["flash"] = "Dismissed and written to memory."
            st.rerun()

    st.write("")
    if st.button("Send Real Slack Notification", key="btn_slack"):
        ok, detail = send_notification(
            CLIENT_ID, SIGNAL_1["signal_text"], "act_this_year",
            signal_id=SIGNAL_1["signal_id"], momentum=SIGNAL_1["hazra_momentum_score"],
            timing_note=TIMING_NOTE, past_decision=PAST_DECISION + " [MEM-HAL-2024-001]",
        )
        (st.success if ok else st.error)(detail)
    st.caption("Buttons in the Slack message reach this app through webhook.py and an ngrok "
               "tunnel (see README). The buttons above always work without them.")


# ---------------------------------------------------------------------------- shell
NAV_ICONS = {  # Material Symbols, rendered by Streamlit's own icon font (no emoji)
    "Main Interaction": ":material/auto_awesome:",
    "Memory View": ":material/history:",
    "Trace Log": ":material/receipt_long:",
    "Proactive Notification": ":material/notifications:",
}

# A widget's session key cannot be assigned after the widget renders, so tab switches
# from inside a tab are staged in switch_tab and applied here, before the radio exists.
if st.session_state.pop("switch_tab", False):
    st.session_state["nav"] = TAB_NAMES[st.session_state["active_tab"]]

with st.sidebar:
    st.markdown(flat("""
    <div style="padding: 20px 16px 16px; border-bottom: 1px solid #1e293b; margin-bottom: 12px;">
        <div class="brand-title" style="font-size: 16px; font-weight: 600;">Recall</div>
        <div class="brand-sub" style="font-size: 11px; margin-top: 2px;">CREWASIS Decision Intelligence</div>
    </div>
    """), unsafe_allow_html=True)
    st.radio("nav", TAB_NAMES, key="nav", label_visibility="collapsed",
             format_func=lambda name: f"{NAV_ICONS[name]}  {name}")

st.markdown(flat(f"""
<div style="display:flex; justify-content:space-between; align-items:center;
     padding-bottom:16px; border-bottom:1px solid #f1f5f9; margin-bottom:20px;">
    <div>
        <span style="font-size:20px; font-weight:600; color:#0f172a;">Recall</span>
        <span style="font-size:13px; color:#94a3b8; margin-left:12px;">CREWASIS | {esc(CLIENT_ID.title())} | GLP-1 Nutrition</span>
    </div>
    <div style="display:flex; align-items:center; gap:8px;">
        <span style="width:7px; height:7px; background:#16a34a; border-radius:50%; display:inline-block;"></span>
        <span style="font-size:12px; color:#64748b;">Live</span>
        <span style="font-size:12px; background:#f1f5f9; color:#475569; padding:3px 10px;
              border-radius:20px; border:1px solid #e2e8f0;">Enterprise</span>
    </div>
</div>
"""), unsafe_allow_html=True)

if _toast := st.session_state.pop("toast", None):
    st.toast(_toast)

{
    TAB_NAMES[0]: tab_main,
    TAB_NAMES[1]: tab_memory,
    TAB_NAMES[2]: tab_trace,
    TAB_NAMES[3]: tab_notification,
}[st.session_state["nav"]]()


# ------------------------------------------------------------------- slack polling
# A button clicked in Slack lands in pending_queries via webhook.py. This picks it up.
#
# It is an st.fragment on a timer, not sleep + st.rerun(): a fragment rerun redraws only
# itself, and this one draws nothing, so the page never flickers or reloads while waiting.
# The timer exists only while the pipeline is idle. Polling at a human gate would rerun
# the app underneath the approval buttons. It sits at the end of the script so the idle
# check reflects the state AFTER this run.
#
# "Idle" is read from the checkpointed graph, not from a flag in session_state: a flag can
# be left stuck by an interrupted run and would then silence Slack for good.
ABANDONED_RUN_SECONDS = 90


def pipeline_idle() -> bool:
    config = st.session_state.get("run_config")
    if not config:
        return True                                   # nothing has run yet
    snap = graph.get_state(config)
    if not snap.next:
        return True                                   # finished
    if any(task.interrupts for task in snap.tasks):
        return False                                  # paused at a human gate
    try:                                              # mid-run, unless it was abandoned
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(snap.created_at)).total_seconds()
    except (TypeError, ValueError):
        return False
    return age > ABANDONED_RUN_SECONDS


@st.fragment(run_every="2s" if pipeline_idle() else None)
def slack_poller():
    if not pipeline_idle():
        return
    row = claim_pending_query(conn, CLIENT_ID)
    if row:
        st.session_state["auto_query"] = row["query"]
        st.session_state["active_tab"] = 0
        st.session_state["switch_tab"] = True
        st.session_state["toast"] = "Slack: Explore This Now clicked. Running the pipeline."
        st.rerun()  # scope="app": leave the fragment, redraw the page, run the pipeline


slack_poller()
