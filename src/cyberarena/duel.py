"""Interleaved red/blue duel (Phase B).

The Red Team agent runs step-by-step; after each batch of red tool calls the
Blue Team defender inspects the fresh logs and may apply real defensive
actions (block_ip / isolate_host / reset_credentials) that immediately remove
red-team capability. The duel outcome is measured from simulation state:
success means the attacker ACTUALLY exfiltrated data, not merely that it
claimed completion via mark_task_complete.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import config
from .blue_team.defender import run_defender_turn
from .red_team.agent import build_red_agent
from .sim.log_store import render_line
from .sim.runtime import Simulation
from .sim.scenarios import Scenario

RED_START_MSG = "Begin the operation and pursue the objective."


@dataclass
class DuelResult:
    scenario_key: str
    scenario_name: str
    red_success: bool
    red_done: bool
    exfiltrated_hosts: list = field(default_factory=list)
    exfil_attempts_dropped: int = 0
    red_moves: int = 0
    defender_turns: int = 0
    defenses: list = field(default_factory=list)
    blocked_ips: list = field(default_factory=list)
    isolated_hosts: list = field(default_factory=list)
    n_logs: int = 0
    red_trail: str = ""
    elapsed: float = 0.0


def _trail_line(tc: dict) -> str:
    args = ", ".join(f"{k}={v}" for k, v in tc.get("args", {}).items())
    return f"-> {tc['name']}({args})"


def run_duel(
    scenario: Scenario,
    llm=None,
    *,
    red_agent=None,
    defender_fn=None,
    max_red_steps: int | None = None,
    defender_every: int = 1,
) -> DuelResult:
    """Run one interleaved duel and return the measured outcome.

    `red_agent` and `defender_fn` are injectable for deterministic tests:
      - red_agent: an agent exposing .stream(), or a factory(sim) -> agent.
      - defender_fn: callable(sim) -> list of applied-action dicts.
    When omitted they default to the LLM red agent and LLM defender.
    """
    t0 = time.time()
    sim = Simulation(scenario)
    max_red_steps = max_red_steps or config.MAX_RED_STEPS

    if red_agent is None:
        from .llm_factory import get_chat_model

        llm = llm or get_chat_model()
        agent = build_red_agent(llm, sim, scenario, max_steps=max_red_steps)
    else:
        agent = red_agent(sim) if callable(red_agent) else red_agent

    if defender_fn is None:
        from .llm_factory import get_chat_model

        llm = llm or get_chat_model()
        context = sim.net.network_summary()
        seen = {"n": 0}

        def defender_fn(s):  # type: ignore[misc]
            entries = s.logs.snapshot()
            new_lines = [render_line(e) for e in entries[seen["n"]:]]
            seen["n"] = len(entries)
            turn = run_defender_turn(llm, s, context=context, new_lines=new_lines)
            return turn["actions"]

    trail: list[str] = []
    defenses: list = []
    red_moves = 0
    defender_turns = 0
    last_log_count = sim.total_logs()

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": RED_START_MSG}]},
        config={"recursion_limit": max_red_steps * 4},
        stream_mode="updates",
    ):
        for msg in chunk.get("agent", {}).get("messages", []):
            for tc in getattr(msg, "tool_calls", None) or []:
                trail.append(_trail_line(tc))
        if "tools" not in chunk:
            continue
        red_moves += 1
        if sim.state["done"]:
            break
        has_new_logs = sim.total_logs() > last_log_count
        if not has_new_logs or red_moves % defender_every:
            last_log_count = sim.total_logs()
            continue
        defender_turns += 1
        defenses.extend(defender_fn(sim) or [])
        # Advance the watermark AFTER the defender turn so the defender's own
        # defensive_action logs are not mistaken for new attacker activity.
        last_log_count = sim.total_logs()

    return DuelResult(
        scenario_key=scenario.key,
        scenario_name=scenario.name,
        red_success=bool(sim.state["exfiltrated"]),
        red_done=bool(sim.state["done"]),
        exfiltrated_hosts=sorted(sim.state["exfiltrated"]),
        exfil_attempts_dropped=int(sim.state["exfil_dropped"]),
        red_moves=red_moves,
        defender_turns=defender_turns,
        defenses=defenses,
        blocked_ips=sorted(sim.state["blocked_ips"]),
        isolated_hosts=sorted(sim.state["isolated_hosts"]),
        n_logs=sim.total_logs(),
        red_trail="\n".join(trail) if trail else "(no tool calls recorded)",
        elapsed=time.time() - t0,
    )


def duel_to_dict(r: DuelResult) -> dict:
    return {
        "scenario_key": r.scenario_key,
        "scenario_name": r.scenario_name,
        "red_success": r.red_success,
        "red_done": r.red_done,
        "exfiltrated_hosts": r.exfiltrated_hosts,
        "exfil_attempts_dropped": r.exfil_attempts_dropped,
        "red_moves": r.red_moves,
        "defender_turns": r.defender_turns,
        "defenses": r.defenses,
        "blocked_ips": r.blocked_ips,
        "isolated_hosts": r.isolated_hosts,
        "n_logs": r.n_logs,
        "red_trail": r.red_trail,
        "elapsed_seconds": round(r.elapsed, 2),
    }


# --- Phase C: batch duel evaluation + statistics ---------------------------


def summarize_duels(results: list[DuelResult]) -> dict:
    """Aggregate duel outcomes with confidence intervals."""
    from .eval.metrics import mean_ci, wilson_ci

    n = len(results)
    wins = sum(1 for r in results if r.red_success)
    claims = sum(1 for r in results if r.red_done)
    dropped = sum(r.exfil_attempts_dropped for r in results)

    action_counts: dict[str, int] = {}
    for r in results:
        for d in r.defenses:
            kind = d.get("action", "other") if isinstance(d, dict) else "other"
            action_counts[kind] = action_counts.get(kind, 0) + 1

    return {
        "n_duels": n,
        "red_wins": wins,
        "red_win_rate": wilson_ci(wins, n),
        "red_claim_rate": wilson_ci(claims, n),
        "red_moves": mean_ci([float(r.red_moves) for r in results]),
        "defender_turns": mean_ci([float(r.defender_turns) for r in results]),
        "defense_actions": action_counts,
        "defense_actions_total": sum(action_counts.values()),
        "exfil_attempts_dropped": dropped,
        "elapsed_seconds": mean_ci([r.elapsed for r in results]),
    }


def _fmt_ci(ci: dict, pct: bool = True) -> str:
    if pct:
        return f"{ci['point'] * 100:.1f}% [{ci['low'] * 100:.1f}, {ci['high'] * 100:.1f}] (n={ci['n']})"
    return f"{ci['point']:.2f} [{ci['low']:.2f}, {ci['high']:.2f}] (n={ci['n']})"


def format_duel_summary(s: dict) -> str:
    actions = " ".join(f"{k}={v}" for k, v in sorted(s["defense_actions"].items()))
    lines = [
        f"Duels: {s['n_duels']}",
        "",
        "== OUTCOMES ==",
        f"red wins      : {s['red_wins']}/{s['n_duels']}",
        f"red win rate  : {_fmt_ci(s['red_win_rate'])}",
        f"red claim rate: {_fmt_ci(s['red_claim_rate'])}  "
        "(red called mark_task_complete, even when it had not actually won)",
        "",
        "== DEFENSE ==",
        f"defender turns/duel : {_fmt_ci(s['defender_turns'], pct=False)}",
        f"defense actions     : {actions or 'none'} (total {s['defense_actions_total']})",
        f"exfil attempts dropped: {s['exfil_attempts_dropped']}",
        "",
        "== TEMPO ==",
        f"red moves/duel: {_fmt_ci(s['red_moves'], pct=False)}",
        f"elapsed/duel  : {_fmt_ci(s['elapsed_seconds'], pct=False)}s",
    ]
    if s.get("errors"):
        lines.append(f"\nduels lost to errors: {s['errors']}")
    return "\n".join(lines)


def run_duel_eval(
    n_attack: int = 2,
    runs: int = 1,
    seed: int = 1234,
    archetypes: list[str] | None = None,
    fixed: bool = False,
    llm=None,
    out_dir: str = "outputs",
    retries: int = 1,
    **duel_kwargs,
) -> tuple[list[DuelResult], dict]:
    """Run a reproducible batch of duels and return aggregate statistics.

    fixed=True duels the fixed reference attack scenarios; otherwise generated
    instances are used (n_attack per archetype). Extra kwargs are passed to
    run_duel (e.g. defender_every, or injected red_agent/defender_fn in tests).
    A duel lost to a transient error (network reset, rate limit) is retried
    `retries` times from scratch; persistent failures are counted and skipped.
    """
    import json
    import os
    import sys
    from datetime import datetime, timezone

    from .llm_factory import get_chat_model
    from .sim.scenario_gen import generate_sample
    from .sim.scenarios import all_scenario_keys, get_scenario

    if llm is None:
        llm = get_chat_model()

    if fixed:
        scenarios = [
            get_scenario(k) for k in all_scenario_keys() if get_scenario(k).is_attack
        ]
    else:
        scenarios = generate_sample(
            n_attack_per_archetype=n_attack, n_benign=0, seed=seed,
            archetypes=archetypes,
        )

    results: list[DuelResult] = []
    errors = 0
    for sc in scenarios:
        for _ in range(runs):
            attempt = 0
            while True:
                try:
                    results.append(run_duel(sc, llm, **duel_kwargs))
                    break
                except Exception as exc:
                    attempt += 1
                    if attempt > retries:
                        errors += 1
                        print(
                            f"[duel-eval] duel failed permanently ({sc.key}): {exc}",
                            file=sys.stderr,
                        )
                        break
                    print(
                        f"[duel-eval] duel error ({sc.key}), retrying "
                        f"({attempt}/{retries}): {exc}",
                        file=sys.stderr,
                    )

    summary = summarize_duels(results)
    summary["errors"] = errors

    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"duelval_{stamp}.json")
    payload = {
        "config": {
            "mode": "fixed" if fixed else "generated",
            "n_attack_per_archetype": n_attack, "runs_per_instance": runs,
            "seed": seed, "archetypes": archetypes,
        },
        "summary": summary,
        "duels": [duel_to_dict(r) for r in results],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return results, summary


def format_duel(r: DuelResult) -> str:
    exfil = ", ".join(r.exfiltrated_hosts) if r.exfiltrated_hosts else "none"
    lines = [
        "== DUEL RESULT ==",
        f"scenario      : {r.scenario_key} ({r.scenario_name})",
        f"red success   : {r.red_success} (exfiltrated: {exfil}; "
        f"dropped exfil attempts: {r.exfil_attempts_dropped})",
        f"red done      : {r.red_done}",
        f"red moves     : {r.red_moves}",
        f"defender turns: {r.defender_turns}",
        f"blocked IPs   : {', '.join(r.blocked_ips) or 'none'}",
        f"isolated hosts: {', '.join(r.isolated_hosts) or 'none'}",
        f"logs          : {r.n_logs}",
        f"elapsed       : {r.elapsed:.1f}s",
        "",
        "defenses applied:",
    ]
    if r.defenses:
        for d in r.defenses:
            if isinstance(d, dict):
                lines.append(f"  - {d.get('ts', '?')} {d.get('detail', '')}")
            else:
                lines.append(f"  - {d}")
    else:
        lines.append("  (none)")
    lines += ["", "== RED TRAIL ==", r.red_trail]
    return "\n".join(lines)
