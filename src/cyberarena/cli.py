"""Command-line interface for CyberArena.

Commands:
  list             Show available scenarios.
  run SCENARIO     Run one scenario end-to-end and print the incident report.
  eval [KEYS...]   Run the evaluation harness over scenarios (optionally
                   --runs N, --provider X) and print metrics.
  eval-gen         Run the generated-scenario evaluation (randomized instances)
                   and report metrics with confidence intervals.
  duel SCENARIO    Run an interleaved red/blue duel: the defender acts between
                   red-team moves with real defensive actions.
  duel-eval        Run a batch of duels and report red win rate + defense
                   statistics with confidence intervals.
  injection-eval   Blue Team prompt-injection robustness: identical attack with
                   and without adversarial text embedded in the logs.
"""
from __future__ import annotations

import argparse
import json

from .eval.metrics import benign_metrics, detection_stats, finding_metrics
from .eval.runner import run_eval, run_single_scenario
from .llm_factory import get_chat_model
from .sim.scenarios import all_scenario_keys, get_scenario


def _provider_arg(args) -> str | None:
    return getattr(args, "provider", None) or None


def _format_metrics(results) -> str:
    from .eval.metrics import RunResult

    lines = []
    for r in results:
        head = (
            f"{r.scenario_key} ({r.scenario_name}) "
            f"attack={r.is_attack} detection={r.detection} "
            f"conf={r.triage_confidence:.2f}"
        )
        lines.append(head)
        fm = finding_metrics(r)
        lines.append(
            f"  evidence: precision={fm['precision']:.2f} recall={fm['recall']:.2f} "
            f"f1={fm['f1']:.2f} | flagged={fm['flagged_logs']} "
            f"gt={fm['ground_truth_logs']} | {fm['false_positive_logs']} false-positive log(s)"
        )
        lines.append(
            f"  corpus={r.n_logs} logs | gt phases={list(r.gt_by_type)} | red_done={r.red_done}"
        )
        lines.append(
            "  phase recall: "
            + ", ".join(
                f"{phase}={sum(1 for i in ids if i in r.flagged_ids)}/{len(ids)}"
                for phase, ids in r.gt_by_type.items()
            )
        )
        if hasattr(r, "elapsed"):
            lines.append(f"  elapsed={r.elapsed:.1f}s")  # type: ignore[attr-defined]
        lines.append("")
    stats = detection_stats([r for r in results if isinstance(r, RunResult)])
    lines.append("== DETECTION (triage vs ground truth) ==")
    lines.append(
        f"TP={stats['tp']} FP={stats['fp']} TN={stats['tn']} FN={stats['fn']}"
    )
    lines.append(
        f"accuracy={stats['accuracy']:.2f} detection_precision={stats['detection_precision']:.2f} "
        f"FPR={stats['false_positive_rate']:.2f} FNR={stats['false_negative_rate']:.2f}"
    )
    for r in results:
        if isinstance(r, RunResult) and not r.is_attack:
            bm = benign_metrics(r)
            lines.append(
                f"[benign {r.scenario_key}] log FP rate={bm['log_false_positive_rate']:.3f} "
                f"({bm['flagged_logs']}/{bm['corpus_size']} logs)"
            )
    return "\n".join(lines)


def cmd_list(_args) -> int:
    for key in all_scenario_keys():
        s = get_scenario(key)
        print(f"{key:<22} attack={s.is_attack}  {s.name}")
    return 0


def cmd_run(args) -> int:
    llm = get_chat_model(_provider_arg(args))
    res = run_single_scenario(args.scenario, llm)
    print(f"Scenario : {res.scenario_key} ({res.scenario_name})")
    print(f"Attack   : {res.is_attack}")
    print(f"Detection: {res.detection} (triage confidence {res.triage_confidence:.2f})")
    print(f"Corpus   : {res.n_logs} logs | ground truth logs: {len(res.gt_ids)}")
    print(f"Flagged  : {len(res.flagged_ids)} log(s)")
    print()
    print("== RED TEAM TOOL TRAIL ==")
    print(res.red_trail or "(none)")
    print()
    print("== INCIDENT REPORT ==")
    print(res.report)
    return 0


def cmd_eval(args) -> int:
    keys = args.keys or all_scenario_keys()
    llm = get_chat_model(_provider_arg(args))
    results = run_eval(keys=keys, runs=args.runs, llm=llm)
    print(_format_metrics(results))
    print("\n(output JSON written to outputs/)")
    return 0


def cmd_eval_gen(args) -> int:
    from .eval.metrics import format_summary
    from .eval.runner import run_generated_eval

    llm = get_chat_model(_provider_arg(args))
    results, summary = run_generated_eval(
        n_attack=args.instances,
        n_benign=args.benign,
        runs=args.runs,
        seed=args.seed,
        archetypes=args.archetypes or None,
        llm=llm,
    )
    print(format_summary(summary))
    print("\n(output JSON written to outputs/)")
    return 0


def cmd_duel(args) -> int:
    import os
    from datetime import datetime, timezone

    from .duel import duel_to_dict, format_duel, run_duel

    llm = get_chat_model(_provider_arg(args))
    res = run_duel(
        get_scenario(args.scenario),
        llm,
        max_red_steps=args.max_red_steps,
        defender_every=args.defender_every,
    )
    print(format_duel(res))

    os.makedirs("outputs", exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join("outputs", f"duel_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(duel_to_dict(res), f, indent=2, default=str)
    print(f"\n(output JSON written to {path})")
    return 0


def cmd_duel_eval(args) -> int:
    from .duel import format_duel_summary, run_duel_eval

    llm = get_chat_model(_provider_arg(args))
    duel_kwargs = {}
    if getattr(args, "max_red_steps", None):
        duel_kwargs["max_red_steps"] = args.max_red_steps
    results, summary = run_duel_eval(
        n_attack=args.instances,
        runs=args.runs,
        seed=args.seed,
        archetypes=args.archetypes or None,
        fixed=args.fixed,
        llm=llm,
        defender_every=args.defender_every,
        **duel_kwargs,
    )
    print(format_duel_summary(summary))
    print("\n(output JSON written to outputs/)")
    return 0


def cmd_injection_eval(args) -> int:
    from .eval.injection import format_injection, run_injection_eval

    llm = get_chat_model(_provider_arg(args))
    summary, _results = run_injection_eval(runs=args.runs, llm=llm)
    print(format_injection(summary))
    print("\n(output JSON written to outputs/)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cyberarena",
        description="Multi-agent AI cyber defense simulation (Stage 1).",
    )
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini", "mock"])
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="list scenarios").set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="run one scenario end-to-end")
    p_run.add_argument("scenario", choices=all_scenario_keys())
    p_run.add_argument("--provider", choices=["openai", "anthropic", "gemini", "mock"])
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("eval", help="run evaluation harness")
    p_eval.add_argument("keys", nargs="*", help="scenario keys (default: all)")
    p_eval.add_argument("--runs", type=int, default=1)
    p_eval.add_argument("--provider", choices=["openai", "anthropic", "gemini", "mock"])
    p_eval.set_defaults(func=cmd_eval)

    p_gen = sub.add_parser(
        "eval-gen", help="run generated-scenario evaluation with confidence intervals"
    )
    p_gen.add_argument(
        "--instances", type=int, default=10, help="attack instances per archetype"
    )
    p_gen.add_argument("--benign", type=int, default=5, help="number of benign instances")
    p_gen.add_argument("--runs", type=int, default=1, help="runs per instance")
    p_gen.add_argument("--seed", type=int, default=1234, help="generator seed")
    p_gen.add_argument(
        "--archetypes", nargs="*",
        choices=["phishing_lateral", "brute_escalation"],
        help="attack archetypes to generate (default: both)",
    )
    p_gen.add_argument("--provider", choices=["openai", "anthropic", "gemini", "mock"])
    p_gen.set_defaults(func=cmd_eval_gen)

    p_duel = sub.add_parser("duel", help="run interleaved red/blue duel")
    p_duel.add_argument(
        "scenario",
        choices=[k for k in all_scenario_keys() if get_scenario(k).is_attack],
    )
    p_duel.add_argument("--max-red-steps", type=int, default=None)
    p_duel.add_argument(
        "--defender-every", type=int, default=1,
        help="run a defender turn every N red moves (default: every move)",
    )
    p_duel.add_argument("--provider", choices=["openai", "anthropic", "gemini", "mock"])
    p_duel.set_defaults(func=cmd_duel)

    p_deval = sub.add_parser(
        "duel-eval", help="run batch duel evaluation with win-rate statistics"
    )
    p_deval.add_argument(
        "--instances", type=int, default=2,
        help="generated attack instances per archetype (ignored with --fixed)",
    )
    p_deval.add_argument("--runs", type=int, default=1, help="duels per instance")
    p_deval.add_argument("--seed", type=int, default=1234, help="generator seed")
    p_deval.add_argument(
        "--archetypes", nargs="*",
        choices=["phishing_lateral", "brute_escalation"],
        help="attack archetypes to generate (default: both)",
    )
    p_deval.add_argument(
        "--fixed", action="store_true",
        help="duel the fixed reference attack scenarios instead of generated ones",
    )
    p_deval.add_argument("--defender-every", type=int, default=1)
    p_deval.add_argument(
        "--max-red-steps", type=int, default=None,
        help="tool-call budget for the red agent (default: MAX_RED_STEPS env)",
    )
    p_deval.add_argument("--provider", choices=["openai", "anthropic", "gemini", "mock"])
    p_deval.set_defaults(func=cmd_duel_eval)

    p_inj = sub.add_parser(
        "injection-eval",
        help="Blue Team prompt-injection robustness eval (clean vs injected)",
    )
    p_inj.add_argument("--runs", type=int, default=3, help="Blue runs per arm")
    p_inj.add_argument("--provider", choices=["openai", "anthropic", "gemini", "mock"])
    p_inj.set_defaults(func=cmd_injection_eval)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())