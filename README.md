# Recall: CREWASIS Hackathon 2026

Streamlit + LangGraph prototype of the Recall multi-agent pipeline for Haleon's GLP-1
vitamins and supplements team. Synthetic HAZRA signals, SQLite client memory, three
real human approval gates, and a full agent action log.

## Run

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in the keys
python -m scripts.tune_thresholds   # optional, offline, no keys needed
streamlit run app.py
```

Python 3.12 is recommended: PyTorch wheels for the newest interpreters lag behind.
The first run downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB).

## Architecture

Three layers: data at the bottom, four agents in the middle, humans on top with a
mandatory gate at every consequential decision.

```
router (pure Python)
  -> memory agent -+-> [HUMAN: conflict gate] -> strategy agent
                   |     retrieve -> assess <-> secondary_retrieve -> draft + validate
                   |                                   -> [HUMAN: brief approval] -> memory_write
                   +-> claims agent
                         retrieve -> draft -> flag against memory
                                   -> [HUMAN: claim selection] -> memory_write
```

- **One LangGraph graph**, typed `RecallState`, compiled with `SqliteSaver` on
  `recall.db`. The bracketed nodes call `langgraph.types.interrupt()`, so the graph
  genuinely stops and can only continue when the same `thread_id` is resumed with
  `Command(resume=...)`. Approval state survives a server restart.
- **`memory_write` is downstream of every gate.** Nothing reaches `client_memory`
  without a human action having resumed the graph.
- **No model call in routing or validation.** The router and citation validator are
  pure Python. Every LLM call uses structured output and a three-attempt retry
  (0s, 2s, 4s), and on total failure the analyst sees a message, never a traceback.
- **Hybrid retrieval.** BM25 and dense (MiniLM cosine) run in parallel and are merged
  with reciprocal rank fusion, k=60.
- **Citation enforcement.** Every claim must carry `[SOURCE: signal_id]`. Validation
  checks for uncited claims, ghost citations (ids not retrieved this session) and a
  0.50 confidence floor. A failure re-prompts the model with the specific errors named,
  up to three times, then surfaces a structured error. The analyst never sees an
  ungrounded brief.
- **Observability.** LangSmith traces every graph run and completion. Every agent
  action is also written to `agent_action_log` (Tab 3).

## The three demo queries

1. *"What should we do about the muscle health opportunity for GLP-1 customers?"*
   Memory finds precedent MEM-HAL-2024-001 (yellow banner). Signals 001, 002, 003,
   confidence 0.875. Approve.
2. *"Draft a marketing claim for our GLP-1 muscle preservation product."*
   Three options; the "clinically proven" option is flagged red against
   MEM-HAL-2024-002 with the legal reason. Reject it, select a safe one.
3. *"What is the competitive whitespace in GLP-1 nutrition for Haleon?"*
   Memory raises a conflict and the conflict gate stops the run until the analyst
   decides. The strategy agent then finds no consumer-language signal, names the gap,
   runs a targeted second retrieval that pulls HAZ-2024-GLP1-004, and reassesses. The
   loop is visible in the state panel.

## Deviations from the PRD, all deliberate

1. **Confidence formula.** The PRD's `mean(confidence * momentum)` returns 0.749 for the
   Query 1 signals: amber, and below the 0.75 the demo requires. `compute_confidence`
   uses the momentum-weighted average instead (0.875). The literal version is kept as
   `compute_confidence_prd_literal` for comparison.
2. **Secondary retrieval trigger.** The PRD triggers on confidence below 0.70. Because
   confidence is a weighted average of signals that each score 0.79 to 0.94, it can
   essentially never fall that low. The trigger is a coverage gap instead: a whitespace
   question with no `consumer_language_analysis` signal retrieved. It is deterministic
   and explains itself.
3. **Memory thresholds and the positioning rule.** `scripts/tune_thresholds.py` showed
   Query 1 scoring 0.77 against the rejected muscle claim, higher than Query 3 (0.41), so
   no similarity threshold can make Query 3 the conflict and Query 1 the precedent. The
   PRD's thresholds (conflict 0.80, precedent 0.60) are used, plus one explicit rule:
   for a whitespace or positioning question, any related rejected decision (similarity
   0.35 or more) is a conflict. Whitespace in HAZ-2024-GLP1-002 is a clinical-language
   claim, which is the direction legal rejected. This rule is calibrated to the demo
   and should be replaced by an LLM-judged check in production.
4. **Claim flagging ignores embedding similarity.** Measured on this data, a safe claim
   scores 0.91 against the rejected claim, the same as a risky paraphrase, because
   embeddings capture topic, not hedging. Flagging uses string similarity (0.70) and the
   risky phrases from HAZ-2024-GLP1-005.
5. **One connection per role.** The checkpointer opens its own connection to
   `recall.db` rather than sharing the app's, so LangGraph's writes and the app's reads
   never contend on one cursor.
6. **`interrupt()` only, no `interrupt_before`.** Using both pauses the graph twice at
   each gate.
7. **Tabs are a radio control, not `st.tabs`.** Streamlit cannot switch `st.tabs`
   programmatically, and Tab 4's "Explore This Now" must jump to Tab 1 and run.
8. **`langgraph-checkpoint-sqlite` is added to requirements.** `SqliteSaver` lives in
   that separate package.

## Structured output on Azure

The client first requests `json_schema` (server-enforced). If the deployment rejects
that response format it drops to `json_object` and checks required fields and enums
itself. The downgrade is never silent: the mode is written to `agent_action_log`
(`model_used`) and shown in Tab 3.

## Known gaps

- **Slack buttons need two extra processes.** They work through `webhook.py` (FastAPI, port 8000)
  and an ngrok tunnel: Slack POSTs the click to the tunnel, the webhook verifies the signing
  secret and writes to SQLite, and the Streamlit app picks it up within about 2 seconds.
  Run `./venv/bin/uvicorn webhook:app --port 8000` and `ngrok http 8000`, then set the
  Interactivity Request URL to `https://<ngrok-host>/slack/actions`. Explore runs the pipeline;
  Hold and Not Relevant write to memory. The polling pauses while a brief is awaiting approval
  and any click waits for the next idle moment. A click older than 10 minutes expires.
- **The claims agent runs directly on claims queries** rather than after a brief
  approval, matching the three demo scenarios.
- **`recall.db` is local.** Delete it to reset memory to the three seed decisions.
