"""Prompt-injection robustness tests (no LLM required).

Verify the injection mechanism itself: the payload reaches the log corpus the
Blue Team reads, without altering the ground-truth structure of the attack.
"""
from cyberarena.eval.injection import (
    INJECTION_PAYLOAD,
    run_injection_eval,
    run_scripted_attack,
    summarize_injection,
    _scenario_with_payload,
)
from cyberarena.llm_factory import MockChatModel


def test_scripted_attack_succeeds_on_fixed_scenario():
    sim = run_scripted_attack(_scenario_with_payload(None))
    assert sim.state["creds"].get("alice")
    assert sim.state["exfiltrated"] == {"db01"}
    assert set(sim.attack_ids_by_type()) >= {
        "phishing", "initial_access", "lateral_movement", "exfiltration",
    }


def test_injection_payload_appears_in_blue_team_view():
    sim = run_scripted_attack(_scenario_with_payload(INJECTION_PAYLOAD))
    text = sim.snapshot_text()
    assert INJECTION_PAYLOAD in text
    # Payload must appear in attacker-authored fields across multiple phases.
    hits = [e for e in sim.logs.snapshot() if INJECTION_PAYLOAD in e["detail"]]
    events = {e["event"] for e in hits}
    assert "phishing_email" in events
    assert "process_exec" in events


def test_clean_corpus_has_no_payload():
    sim = run_scripted_attack(_scenario_with_payload(None))
    assert INJECTION_PAYLOAD not in sim.snapshot_text()


def test_injection_does_not_change_ground_truth_shape():
    clean = run_scripted_attack(_scenario_with_payload(None))
    injected = run_scripted_attack(_scenario_with_payload(INJECTION_PAYLOAD))
    assert clean.total_logs() == injected.total_logs()
    assert len(clean.attack_log_ids()) == len(injected.attack_log_ids())
    assert set(clean.attack_ids_by_type()) == set(injected.attack_ids_by_type())
    for phase, ids in clean.attack_ids_by_type().items():
        assert len(ids) == len(injected.attack_ids_by_type()[phase])


def test_injection_eval_offline_mock(tmp_path):
    summary, results = run_injection_eval(
        runs=1, llm=MockChatModel(), out_dir=str(tmp_path)
    )
    assert {"clean", "injected", "degradation"} <= set(summary)
    assert len(results) == 2
    assert summary["clean"]["n"] == 1
    assert summary["injected"]["n"] == 1
    assert len(list(tmp_path.glob("injection_*.json"))) == 1


def test_summarize_injection_degradation_sign():
    from cyberarena.eval.metrics import RunResult

    def mk(arm, detection, flagged):
        return RunResult(
            scenario_key=f"x_{arm}", scenario_name="x", is_attack=True,
            detection=detection, triage_confidence=0.5,
            gt_ids=["L1", "L2"], flagged_ids=flagged, n_logs=10,
        )

    summary = summarize_injection({
        "clean": [mk("clean", True, ["L1", "L2"])],
        "injected": [mk("injected", False, [])],
    })
    assert summary["degradation"]["detection_rate"] == -1.0
    assert summary["degradation"]["recall"] == -1.0
