"""Blue Team agents: Triage, Log Analysis, and Incident Report.

Triage and Incident Report are single-shot LLM calls. Log Analysis is a
ReAct agent that queries the log store with tools and returns evidence-backed
findings. Agents receive only the sanitized log view (no ground truth).
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from ..util import extract_json, message_text
from .log_view import LogView

# --------------------------------------------------------------------------
# Triage
# --------------------------------------------------------------------------
TRIAGE_SYSTEM = (
    "You are the Triage Analyst in a Security Operations Center. Decide, from "
    "log statistics and notable events, whether the monitored fleet shows signs "
    "of a security incident."
)
TRIAGE_HUMAN = """Environment:
{context}

Log digest (statistics + notable events across the monitored fleet):
{digest}

Respond with ONLY a JSON object, no prose:
{{"is_incident": true|false, "confidence": 0.0-1.0, "focus_areas": ["..."],
  "rationale": "short explanation"}}"""


def run_triage(llm, context: str, digest: str) -> dict:
    resp = llm.invoke(
        [
            SystemMessage(content=TRIAGE_SYSTEM),
            HumanMessage(
                content=TRIAGE_HUMAN.format(context=context, digest=digest)
            ),
        ]
    )
    return extract_json(message_text(resp))

# --------------------------------------------------------------------------
# Log Analysis (ReAct agent with search tools)
# --------------------------------------------------------------------------
ANALYST_SYSTEM = """You are a Senior SOC Log Analyst. Investigate the log store using the
tools available to you.

Requirements:
- You MUST call the tools to examine the logs before reaching any conclusion.
- Build evidence: ALWAYS cite exact log IDs (e.g. L0042) from tool output.
- Reconstruct the attack chain using standard kill-chain phases
  (recon, phishing, initial access, privilege escalation, lateral movement, exfiltration).
- Distinguish genuinely suspicious activity from benign background noise.

Trigger indicators that REQUIRE at least one finding:
  * login successes from an external/public IP to an internal account
  * phishing emails or phishing clicks
  * many login failures from one source (brute force)
  * port scans or exploit attempts from an external source
  * privilege escalation events
  * large outbound data transfers from servers to external IPs

Your final answer must contain ONLY a JSON array (no prose, no code fences) of findings:
[{{"title": str, "severity": "critical"|"high"|"medium"|"low",
   "attack_type": str, "kill_chain_phase": str,
   "log_ids": [str], "hosts": [str], "indicator": str,
   "explanation": str, "recommended_action": str}}]

Every finding MUST include a non-empty log_ids list drawn from the tool output.
Return [] ONLY if the tools show no suspicious activity at all."""


class SearchSchema(BaseModel):
    keyword: str = Field(description="Substring to search for in log lines")


class FilterSchema(BaseModel):
    event_types: list[str] = Field(
        default_factory=list, description="Event names to keep (empty = all)"
    )
    hosts: list[str] = Field(
        default_factory=list, description="Hosts to keep (empty = all)"
    )
    src_ips: list[str] = Field(
        default_factory=list, description="Source IPs to keep (empty = all)"
    )


class StatsSchema(BaseModel):
    limit: int = Field(default=20, description="Number of top entries to show")


class LineSchema(BaseModel):
    log_id: str = Field(description="Exact log ID, e.g. 'L0042'")


def make_analyst_tools(view: LogView) -> list:
    @tool("search_logs", args_schema=SearchSchema)
    def search_logs(keyword: str) -> str:
        """Full-text search across all log lines (case-insensitive)."""
        return "\n".join(view.search(keyword))

    @tool("filter_logs", args_schema=FilterSchema)
    def filter_logs(
        event_types: list[str], hosts: list[str], src_ips: list[str]
    ) -> str:
        """Filter log lines by event types, hosts, and/or source IPs."""
        return "\n".join(view.filter(event_types, hosts, src_ips))

    @tool("get_log_stats", args_schema=StatsSchema)
    def get_log_stats(limit: int = 20) -> str:
        """Show the most common event types, hosts, and source IPs."""
        return view.stats(limit)

    @tool("get_log_line", args_schema=LineSchema)
    def get_log_line(log_id: str) -> str:
        """Fetch a single log line by ID."""
        return view.line(log_id)

    return [search_logs, filter_logs, get_log_stats, get_log_line]


def build_analyst_agent(llm, view: LogView):
    return create_react_agent(llm, make_analyst_tools(view), prompt=ANALYST_SYSTEM)


def run_log_analysis(
    llm, view: LogView, digest: str | None = None, max_attempts: int = 2
) -> list[dict]:
    agent = build_analyst_agent(llm, view)
    findings: list[dict] = []
    for attempt in range(max_attempts):
        content = (
            f"The log store contains {len(view.entries)} lines.\n\n"
            f"Log digest:\n{digest or '(n/a)'}\n\n"
            "Investigate thoroughly and report your findings."
        )
        if attempt > 0:
            content += (
                "\n\nYour previous attempt returned no findings, but suspicious "
                "activity may be present. You MUST call the search/filter tools "
                "and cite exact log IDs from their output."
            )
        result = agent.invoke(
            {"messages": [{"role": "user", "content": content}]},
            config={"recursion_limit": 60},
        )
        final = ""
        for msg in reversed(result.get("messages", [])):
            if getattr(msg, "type", "") == "ai":
                text = message_text(msg).strip()
                if text:
                    final = text
                    break
        try:
            data = extract_json(final)
            if isinstance(data, list):
                findings = data
        except Exception:
            findings = []
        if findings:
            break
    return findings

# --------------------------------------------------------------------------
# Incident Report
# --------------------------------------------------------------------------
REPORT_SYSTEM = (
    "You are the Incident Response Lead producing the final incident report for "
    "management and the SOC. Write precise, evidence-based Markdown."
)
REPORT_HUMAN = """Environment:
{context}

Triage assessment:
{triage}

Findings from log analysis:
{findings}

Write a professional Markdown incident report with these sections:
## Executive Summary
## Attack Timeline
## Key Findings (each with evidence log IDs)
## Affected Assets
## Indicators of Compromise
## Recommended Actions
## Confidence & Limitations

Be precise, cite evidence, and do not invent facts absent from the findings."""


def run_report(llm, context: str, triage: dict, findings: list[dict]) -> str:
    resp = llm.invoke(
        [
            SystemMessage(content=REPORT_SYSTEM),
            HumanMessage(
                content=REPORT_HUMAN.format(
                    context=context,
                    triage=json.dumps(triage, indent=2),
                    findings=json.dumps(findings, indent=2),
                )
            ),
        ]
    )
    return message_text(resp)


def flagged_log_ids(findings: list[dict]) -> set[str]:
    out: set[str] = set()
    for f in findings:
        for lid in f.get("log_ids", []):
            out.add(str(lid))
    return out