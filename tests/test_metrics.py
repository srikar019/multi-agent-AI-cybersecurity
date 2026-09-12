"""Unit tests for evaluation metrics math."""
import pytest

from cyberarena.eval.metrics import RunResult, benign_metrics, detection_stats, finding_metrics


def _res(attack, detection, gt=None, flagged=None, n_logs=100):
    return RunResult(
        scenario_key="s",
        scenario_name="S",
        is_attack=attack,
        detection=detection,
        triage_confidence=0.9 if detection else 0.1,
        gt_ids=gt or [],
        gt_by_type={},
        flagged_ids=flagged or [],
        n_logs=n_logs,
    )


def test_detection_stats_perfect():
    stats = detection_stats(
        [
            _res(True, True),      # TP
            _res(True, True),      # TP
            _res(False, False),    # TN
            _res(False, False),    # TN
        ]
    )
    assert stats["tp"] == 2 and stats["tn"] == 2
    assert stats["accuracy"] == 1.0
    assert stats["false_positive_rate"] == 0.0
    assert stats["false_negative_rate"] == 0.0


def test_detection_stats_false_positives():
    stats = detection_stats(
        [
            _res(True, True),      # TP
            _res(False, True),     # FP
            _res(False, False),    # TN
            _res(True, False),     # FN
        ]
    )
    assert stats["tp"] == 1 and stats["fp"] == 1
    assert stats["tn"] == 1 and stats["fn"] == 1
    assert stats["accuracy"] == 0.5
    assert stats["false_positive_rate"] == 0.5
    assert stats["false_negative_rate"] == 0.5


def test_finding_metrics_precision_recall():
    r = _res(True, True, gt=["L1", "L2", "L3"], flagged=["L1", "L2", "L9", "L10"])
    m = finding_metrics(r)
    assert m["precision"] == 0.5
    assert m["recall"] == 2 / 3
    assert m["f1"] == pytest.approx(2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    assert m["false_positive_logs"] == 2


def test_finding_metrics_no_flagged():
    r = _res(True, True, gt=["L1"], flagged=[])
    m = finding_metrics(r)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_benign_metrics():
    r = _res(False, False, flagged=["L1", "L2"], n_logs=100)
    m = benign_metrics(r)
    assert m["flagged_logs"] == 2
    assert m["log_false_positive_rate"] == 0.02