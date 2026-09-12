"""Ground-truth evaluation metrics for the Blue Team pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class RunResult:
    scenario_key: str
    scenario_name: str
    is_attack: bool
    detection: bool
    triage_confidence: float
    gt_ids: list[str] = field(default_factory=list)
    gt_by_type: dict[str, list[str]] = field(default_factory=dict)
    flagged_ids: list[str] = field(default_factory=list)
    n_logs: int = 0
    red_done: bool = False
    report: str = ""
    red_trail: str = ""


def detection_stats(results: list[RunResult]) -> dict:
    """Detection is the Blue Team's triage decision vs. whether an attack ran."""
    tp = sum(1 for r in results if r.is_attack and r.detection)
    fp = sum(1 for r in results if not r.is_attack and r.detection)
    tn = sum(1 for r in results if not r.is_attack and not r.detection)
    fn = sum(1 for r in results if r.is_attack and not r.detection)
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "n_runs": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "detection_precision": precision,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


def finding_metrics(r: RunResult) -> dict:
    """Evidence-level precision/recall against ground-truth log IDs."""
    gt = set(r.gt_ids)
    flagged = set(r.flagged_ids)
    tp_ids = gt & flagged
    precision = len(tp_ids) / len(flagged) if flagged else 0.0
    recall = len(tp_ids) / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "ground_truth_logs": len(gt),
        "flagged_logs": len(flagged),
        "true_positives": len(tp_ids),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_logs": len(flagged - gt),
    }


def benign_metrics(r: RunResult) -> dict:
    """For benign runs: fraction of the log corpus falsely flagged."""
    n = len(set(r.flagged_ids))
    return {
        "flagged_logs": n,
        "corpus_size": r.n_logs,
        "log_false_positive_rate": (n / r.n_logs) if r.n_logs else 0.0,
    }


# --- statistics: confidence intervals -------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> dict:
    """Wilson score interval for a binomial proportion. Returns low/high/point."""
    if n == 0:
        return {"low": 0.0, "high": 0.0, "point": 0.0, "n": 0}
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return {
        "low": max(0.0, center - half),
        "high": min(1.0, center + half),
        "point": p,
        "n": n,
    }


def mean_ci(values: list[float], z: float = 1.96) -> dict:
    """Normal-approximation CI for a mean. Returns low/high/point."""
    n = len(values)
    if n == 0:
        return {"low": 0.0, "high": 0.0, "point": 0.0, "n": 0}
    m = sum(values) / n
    if n == 1:
        return {"low": m, "high": m, "point": m, "n": 1}
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    se = math.sqrt(var / n)
    return {"low": m - z * se, "high": m + z * se, "point": m, "n": n}


def _fmt_ci(ci: dict, pct: bool = True) -> str:
    if pct:
        return f"{ci['point'] * 100:.1f}% [{ci['low'] * 100:.1f}, {ci['high'] * 100:.1f}] (n={ci['n']})"
    return f"{ci['point']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}] (n={ci['n']})"


def summarize_results(results: list[RunResult]) -> dict:
    """Aggregate detection + evidence metrics with confidence intervals."""
    attacks = [r for r in results if r.is_attack]
    benigns = [r for r in results if not r.is_attack]

    det = detection_stats(results)
    acc_ci = wilson_ci(det["tp"] + det["tn"], det["n_runs"])
    fpr_ci = wilson_ci(det["fp"], det["fp"] + det["tn"])

    prec = mean_ci([finding_metrics(r)["precision"] for r in attacks])
    rec = mean_ci([finding_metrics(r)["recall"] for r in attacks])
    f1 = mean_ci([finding_metrics(r)["f1"] for r in attacks])

    benign_fp = mean_ci([benign_metrics(r)["log_false_positive_rate"] for r in benigns])

    return {
        "n_total": len(results),
        "n_attack": len(attacks),
        "n_benign": len(benigns),
        "detection": {
            "tp": det["tp"], "fp": det["fp"], "tn": det["tn"], "fn": det["fn"],
            "accuracy": acc_ci,
            "false_positive_rate": fpr_ci,
        },
        "evidence": {"precision": prec, "recall": rec, "f1": f1},
        "benign_log_fp_rate": benign_fp,
    }


def format_summary(summary: dict) -> str:
    det = summary["detection"]
    ev = summary["evidence"]
    lines = [
        f"Runs: {summary['n_total']} total "
        f"({summary['n_attack']} attack, {summary['n_benign']} benign)",
        "",
        "== DETECTION (triage vs ground truth) ==",
        f"TP={det['tp']} FP={det['fp']} TN={det['tn']} FN={det['fn']}",
        f"accuracy : {_fmt_ci(det['accuracy'])}",
        f"FPR      : {_fmt_ci(det['false_positive_rate'])}",
        "",
        "== EVIDENCE (attack runs, mean +/- CI) ==",
        f"precision: {_fmt_ci(ev['precision'], pct=False)}",
        f"recall   : {_fmt_ci(ev['recall'], pct=False)}",
        f"f1       : {_fmt_ci(ev['f1'], pct=False)}",
        "",
        f"benign log-level FP rate: {_fmt_ci(summary['benign_log_fp_rate'])}",
    ]
    return "\n".join(lines)