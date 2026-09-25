# Recall
### From intelligence delivered to intelligence that compounds.

Built for the **CREWASIS External Hackathon 2026** by Team 1.

---

## What is Recall

Recall is a multi-agent decision memory layer built on top of CREWASIS's Winston and HAZRA platform. Winston today answers the questions clients bring to it. Recall makes Winston remember every decision, compound every engagement, and proactively surface intelligence before clients even ask.

Every insight, every approved direction, and every rejected claim with its legal reason gets stored as institutional memory. The next engagement starts smarter. Brands stop re-learning what they already paid to learn.

---

## The Problem

Every CREWASIS engagement ends when the answer is delivered. The decision that follows happens somewhere else. The next engagement starts from zero. Years of working with a client and Winston still does not know what they tried, what legal rejected, or what window they are currently in.

---

## The Solution

Three agents working as one pipeline:

**Strategy Drafting Agent:** Takes HAZRA ranked signals and drafts a grounded strategic brief. Every claim has a verified source citation. Confidence scored. Citations validated in Python, not by the model. Secondary retrieval loop if signal is insufficient.

**Claims and Positioning Agent:** Drafts evidence-backed marketing claims ranked by HAZRA consumer resonance. Checks every claim against past legal rejections before the client sees them. Flags matches automatically.

**Memory and Continuity Agent:** Stores every approved brief, selected claim, and rejected direction with reasons as structured institutional memory. Surfaces it when relevant. Next session starts with everything already known.

**Proactive Notification Agent:** HAZRA runs on a scheduled cadence. When a signal crosses a significance threshold, Winston sends a structured brief to Slack or email automatically. Client responds. Memory updates.

---

## Architecture

```
HAZRA weekly schedule
        |
        v
Notification Agent -> Slack / Email -> client responds
        |
        v
Query Router (rules only, no model call)
Signal / Memory / Synthesis / Urgency / Unanswerable
        |
        v
Hybrid Retrieval: BM25 + Dense + Reciprocal Rank Fusion
        |
        v
Citation Enforcement + Ghost Citation Detection (Python)
        |
        v
Orchestrator: routes based on agent findings
        |
        v
Memory Agent: finds conflict or precedent
[HUMAN CONFLICT GATE: LangGraph interrupt]
        |
        v
Strategy Agent: loops until confidence above 0.70
[HUMAN BRIEF GATE: LangGraph interrupt]
        |
        v
Claims Agent: flags past rejections before client sees them
[HUMAN CLAIM GATE: LangGraph interrupt]
        |
        v
Memory Agent writes: atomic transaction, three writes
Next engagement starts smarter
```

---

## Human in the Loop

Three hard LangGraph interrupt gates. The graph cannot advance without a human input event. No timeout. No auto-approve. State checkpointed to SQLite so approval survives server restarts.

- **Conflict gate:** fires before drafting starts if memory finds a conflict
- **Brief approval gate:** analyst refines, client approves or rejects
- **Claim selection gate:** client selects, memory records the decision

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Language |
| Streamlit | UI (four tabs) |
| LangGraph | Multi-agent orchestration with interrupt gates |
| LangSmith | Automatic observability for every agent run |
| Azure GPT-5.4 | LLM, structured output mode only |
| SQLite | Database for memory, logs, and checkpoints |
| BM25 + sentence-transformers | Hybrid retrieval |
| Reciprocal Rank Fusion | Combines BM25 and dense results |
| Slack Block Kit API | Real proactive notifications |
| FastAPI + uvicorn | Webhook server for Slack button responses |
| ngrok | Public tunnel for Slack webhook in development |

---

## Observability

Three layers:

- **LangSmith:** what the agents did and why, full LLM trace per run
- **Datadog:** infrastructure health, API latency, error rates (production)
- **Action log:** what humans decided and what evidence they had, client-facing audit trail

---

## The Urgency Signal

HAZRA captures when a consumer signal first appears and starts climbing. Public launch records give the other end. The gap is the window.

Two verified real-world examples:

- GLP-1 muscle loss: signal detectable early 2022, Herbalife launched February 8 2024 (**24 months**)
- Pet joint health: signal climbing 2020 to 2021, Mars Petcare entered March 2023

Output is an urgency tier: **Act This Quarter / Act This Year / Monitor**. Not a false precision number.

---

## Demo Queries

Three scripted queries that demonstrate the full pipeline:

**Query 1:** "What should we do about the muscle health opportunity for GLP-1 customers?"
Memory agent finds precedent. Brief drafted with three verified citations. Confidence 0.88.

**Query 2:** "Draft a marketing claim for our GLP-1 muscle preservation product."
Claims agent flags clinically proven language automatically from past legal rejection in memory.

**Query 3:** "What is the competitive whitespace in GLP-1 nutrition for Haleon?"
Conflict gate fires before drafting. Secondary retrieval loop visible in step history. Richer brief incorporating four signals and two sessions of memory.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/GauravBohra2001/recall.git
cd recall

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Fill in your credentials in .env

# Run the app
streamlit run app.py
```

For the Slack webhook integration, run in separate terminals:

```bash
uvicorn webhook:app --host 0.0.0.0 --port 8000
ngrok http 8000
```

Then update the Request URL in your Slack app's Interactivity settings.

---

## Environment Variables

```
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT=
AZURE_OPENAI_API_VERSION=
LANGCHAIN_API_KEY=
LANGCHAIN_TRACING_V2=
LANGCHAIN_PROJECT=
CLIENT_ID=
SLACK_BOT_TOKEN=
SLACK_CHANNEL_ID=
SLACK_SIGNING_SECRET=
```

---

## Team

Team 1 - Recall
CREWASIS External Hackathon 2026

- Gaurav Bohra
- Kush Patel
- Nguyen Nguyen (Nicky)

---

*From intelligence delivered to intelligence that compounds.*
