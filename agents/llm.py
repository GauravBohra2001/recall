"""LLM access through Azure AI Foundry.

Structured output only, retry with exponential backoff, never raises to the UI.

Mode handling: the first choice is json_schema (strict, server-enforced). If the
deployment rejects that response_format, the client drops to json_object and checks
the schema itself. That downgrade is never silent: every result carries the `mode`
it was produced under, callers write it to agent_action_log, and it is logged.
The model is never allowed to return freeform text in either mode.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("recall.llm")

RETRY_WAITS_SECONDS = (0, 2, 4)  # attempt 1 immediate, attempt 2 after 2s, attempt 3 after 4s
MAX_COMPLETION_TOKENS = 4000     # generous: reasoning-capable deployments spend these too

MODE_JSON_SCHEMA = "json_schema"
MODE_JSON_OBJECT = "json_object_fallback"

_client = None
_mode = MODE_JSON_SCHEMA
_fallback_reason: Optional[str] = None


@dataclass
class LLMResult:
    parsed: Optional[dict]
    latency_ms: int
    token_count: Optional[int]
    error: Optional[str]
    mode: str

    @property
    def ok(self) -> bool:
        return self.parsed is not None


def deployment() -> str:
    return os.getenv("AZURE_OPENAI_DEPLOYMENT", "")


def model_label(mode: str = None) -> str:
    """What goes in agent_action_log.model_used."""
    return f"{deployment()} [{mode or _mode}]"


def structured_mode_status() -> dict:
    """Surfaced in the UI so a fallback is visible rather than silent."""
    return {"mode": _mode, "fallback_reason": _fallback_reason}


def get_client():
    global _client
    if _client is None:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        )
        try:  # traces every completion in LangSmith under the enclosing graph run
            from langsmith.wrappers import wrap_openai
            client = wrap_openai(client)
        except Exception:  # noqa: BLE001 - tracing must never block a call
            pass
        _client = client
    return _client


def _schema_problems(parsed, schema: dict) -> list:
    """Minimal check for json_object mode, where the server is not enforcing the schema."""
    problems = []
    if not isinstance(parsed, dict):
        return ["response is not a JSON object"]
    spec = schema["schema"]
    for key in spec.get("required", []):
        if key not in parsed:
            problems.append(f"missing field '{key}'")
    for key, prop in spec.get("properties", {}).items():
        if key in parsed and "enum" in prop and parsed[key] not in prop["enum"]:
            problems.append(f"'{key}' must be one of {prop['enum']}")
    return problems


def _is_response_format_rejection(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(t in text for t in ("response_format", "json_schema", "structured output",
                                   "not supported with this model"))


def _build_kwargs(messages, json_schema, mode):
    kwargs = {
        "model": deployment(),
        "messages": messages,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
    }
    if mode == MODE_JSON_SCHEMA:
        kwargs["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    else:
        instruction = (
            "Respond with a single JSON object and nothing else. It must match this "
            "JSON schema exactly:\n" + json.dumps(json_schema["schema"])
        )
        kwargs["messages"] = list(messages) + [{"role": "system", "content": instruction}]
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def call_llm_with_retry(messages, json_schema, max_retries: int = 3) -> LLMResult:
    """Never raises. On total failure returns parsed=None with the last error text."""
    global _mode, _fallback_reason
    last_error = None

    for attempt in range(max_retries):
        wait = RETRY_WAITS_SECONDS[min(attempt, len(RETRY_WAITS_SECONDS) - 1)]
        if wait:
            time.sleep(wait)
        mode = _mode
        try:
            start = time.time()
            response = get_client().chat.completions.create(
                **_build_kwargs(messages, json_schema, mode)
            )
            latency_ms = int((time.time() - start) * 1000)

            content = response.choices[0].message.content
            if not content:
                raise ValueError("empty completion (finish_reason="
                                 f"{response.choices[0].finish_reason})")
            parsed = json.loads(content)

            if mode == MODE_JSON_OBJECT:
                problems = _schema_problems(parsed, json_schema)
                if problems:
                    raise ValueError("schema check failed: " + "; ".join(problems))

            usage = getattr(response, "usage", None)
            return LLMResult(parsed, latency_ms, usage.total_tokens if usage else None,
                             None, mode)

        except Exception as exc:  # noqa: BLE001 - deliberate: nothing may reach the UI
            last_error = f"{type(exc).__name__}: {exc}"
            if mode == MODE_JSON_SCHEMA and _is_response_format_rejection(exc):
                _mode = MODE_JSON_OBJECT
                _fallback_reason = last_error[:300]
                logger.warning("json_schema mode rejected by deployment, falling back to "
                               "json_object with explicit parsing: %s", _fallback_reason)
            else:
                logger.warning("LLM call attempt %d/%d failed: %s",
                               attempt + 1, max_retries, last_error[:300])

    return LLMResult(None, 0, None, last_error, _mode)
