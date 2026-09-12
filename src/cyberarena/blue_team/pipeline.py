"""Blue Team pipeline: Triage -> Log Analysis -> Incident Report.

Wired as a LangGraph StateGraph so later stages (Threat Intel, Root Cause,
Risk Assessment, Correlation) can be inserted between the existing nodes.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from .agents import run_log_analysis, run_report, run_triage


class BlueState(TypedDict):
    scenario_key: str
    context: str
    digest: str
    log_view: object
    triage: dict
    findings: list
    report: str


def build_blue_pipeline(llm):
    def triage_node(state: BlueState) -> dict:
        return {"triage": run_triage(llm, state["context"], state["digest"])}

    def analyze_node(state: BlueState) -> dict:
        return {
            "findings": run_log_analysis(
                llm, state["log_view"], digest=state["digest"]
            )
        }

    def report_node(state: BlueState) -> dict:
        return {
            "report": run_report(
                llm, state["context"], state["triage"], state["findings"]
            )
        }

    g = StateGraph(BlueState)
    g.add_node("triage", triage_node)
    g.add_node("analyze", analyze_node)
    g.add_node("report", report_node)
    g.set_entry_point("triage")
    g.add_edge("triage", "analyze")
    g.add_edge("analyze", "report")
    g.add_edge("report", END)
    return g.compile()


def run_blue_pipeline(llm, *, scenario_key: str, context: str, digest: str, log_view) -> BlueState:
    pipeline = build_blue_pipeline(llm)
    return pipeline.invoke(
        {
            "scenario_key": scenario_key,
            "context": context,
            "digest": digest,
            "log_view": log_view,
        }
    )