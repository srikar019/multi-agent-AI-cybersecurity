"""Evaluation runner: full simulation of Red vs Blue for each scenario."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

from ..blue_team.agents import flagged_log_ids
from ..blue_team.log_view import LogView
from ..blue_team.pipeline import run_blue_pipeline
from ..red_team.agent import run_red_team, red_team_summary
from ..sim.runtime import Simulation
from ..sim.scenarios import Scenario, all_scenario_keys, get_scenario
from .metrics import RunResult, finding_metrics, summarize_results


def run_scenario_obj(scenario: Scenario, llm, *, red_team: bool = True) -> RunResult:
    """Run a single Scenario (fixed or generated) end-to-end."""
    sim = Simulation(scenario)

    red_trail = ""
    if red_team and scenario.is_attack:
        result = run_red_team(sim, scenario, llm)
        red_trail = red_team_summary(result)

    view = LogView(sim.logs.snapshot())
    context = sim.net.network_summary()
    blue = run_blue_pipeline(
        llm,
        scenario_key=scenario.key,
        context=context,
        digest=sim.digest(),
        log_view=view,
    )

    triage = blue.get("triage", {})
    findings = blue.get("findings", [])
    return RunResult(
        scenario_key=scenario.key,
        scenario_name=scenario.name,
        is_attack=scenario.is_attack,
        detection=bool(triage.get("is_incident", False)),
        triage_confidence=float(triage.get("confidence", 0.0) or 0.0),
        gt_ids=sim.attack_log_ids(),
        gt_by_type=sim.attack_ids_by_type(),
        flagged_ids=sorted(flagged_log_ids(findings)),
        n_logs=sim.total_logs(),
        red_done=bool(sim.state.get("done")),
        report=blue.get("report", ""),
        red_trail=red_trail,
    )


def run_single_scenario(key: str, llm, *, red_team: bool = True) -> RunResult:
    return run_scenario_obj(get_scenario(key), llm, red_team=red_team)


def run_eval(
    keys: list[str] | None = None,
    runs: int = 1,
    llm=None,
    out_dir: str = "outputs",
) -> list[RunResult]:
    from ..llm_factory import get_chat_model

    if llm is None:
        llm = get_chat_model()
    keys = keys or all_scenario_keys()

    results: list[RunResult] = []
    for key in keys:
        for i in range(runs):
            t0 = time.time()
            res = run_single_scenario(key, llm)
            res.elapsed = time.time() - t0  # type: ignore[attr-defined]
            results.append(res)

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"eval_{stamp}.json")
    payload = [_result_to_dict(r) for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return results


def run_generated_eval(
    n_attack: int = 10,
    n_benign: int = 5,
    runs: int = 1,
    seed: int = 1234,
    archetypes: list[str] | None = None,
    llm=None,
    out_dir: str = "outputs",
) -> tuple[list[RunResult], dict]:
    """Run a reproducible batch of generated instances and return aggregate stats."""
    from ..llm_factory import get_chat_model
    from ..sim.scenario_gen import generate_sample

    if llm is None:
        llm = get_chat_model()

    scenarios = generate_sample(
        n_attack_per_archetype=n_attack, n_benign=n_benign, seed=seed,
        archetypes=archetypes,
    )

    results: list[RunResult] = []
    errors = 0
    for sc in scenarios:
        for _ in range(runs):
            t0 = time.time()
            try:
                res = run_scenario_obj(sc, llm)
            except Exception as exc:  # quota/network failure: skip, keep batch
                errors += 1
                import sys

                print(f"[eval-gen] run failed ({sc.key}): {exc}", file=sys.stderr)
                continue
            res.elapsed = time.time() - t0  # type: ignore[attr-defined]
            results.append(res)

    summary = summarize_results(results)
    summary["errors"] = errors

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"geneval_{stamp}.json")
    payload = {
        "config": {
            "n_attack_per_archetype": n_attack, "n_benign": n_benign,
            "runs_per_instance": runs, "seed": seed, "archetypes": archetypes,
        },
        "summary": summary,
        "runs": [_result_to_dict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return results, summary


def _result_to_dict(r: RunResult) -> dict:
    fm = finding_metrics(r)
    per_phase_recall = {}
    for phase, ids in r.gt_by_type.items():
        flagged = set(r.flagged_ids)
        hit = sum(1 for i in ids if i in flagged)
        per_phase_recall[phase] = {
            "recall": hit / len(ids) if ids else 0.0,
            "hit": hit,
            "total": len(ids),
        }
    d = {
        "scenario_key": r.scenario_key,
        "scenario_name": r.scenario_name,
        "is_attack": r.is_attack,
        "detection": r.detection,
        "triage_confidence": r.triage_confidence,
        "n_ground_truth_logs": len(r.gt_ids),
        "ground_truth_by_phase": r.gt_by_type,
        "flagged_logs": len(r.flagged_ids),
        "flagged_ids": r.flagged_ids,
        "evidence_metrics": {
            "precision": round(fm["precision"], 3),
            "recall": round(fm["recall"], 3),
            "f1": round(fm["f1"], 3),
            "false_positive_logs": fm["false_positive_logs"],
        },
        "per_phase_recall": per_phase_recall,
        "total_logs": r.n_logs,
        "red_objective_completed": r.red_done,
        "report": r.report,
        "red_trail": r.red_trail,
    }
    if hasattr(r, "elapsed"):
        d["elapsed_seconds"] = round(r.elapsed, 2)  # type: ignore[attr-defined]
    return d