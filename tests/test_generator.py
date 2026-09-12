"""Tests for the scenario generator and evaluation statistics (no LLM)."""
import json
import random

from cyberarena.eval.metrics import (
    RunResult,
    mean_ci,
    summarize_results,
    wilson_ci,
)
from cyberarena.red_team.tools import make_red_tools
from cyberarena.sim import network as net_mod
from cyberarena.sim.runtime import Simulation
from cyberarena.sim.scenario_gen import generate_instance, generate_sample


# --- generator structure ---------------------------------------------------

def test_generate_sample_counts():
    sample = generate_sample(n_attack_per_archetype=3, n_benign=2, seed=1)
    # 2 archetypes x 3 attacks + 2 benign = 8
    assert len(sample) == 8
    attacks = [s for s in sample if s.is_attack]
    benigns = [s for s in sample if not s.is_attack]
    assert len(attacks) == 6 and len(benigns) == 2


def test_generated_models_are_isolated_from_base():
    before = json.dumps(net_mod.HOSTS, sort_keys=True)
    _ = generate_sample(n_attack_per_archetype=2, n_benign=1, seed=5)
    after = json.dumps(net_mod.HOSTS, sort_keys=True)
    assert before == after  # base templates untouched


def test_generated_instances_have_independent_models():
    a = generate_instance("phishing_lateral", random.Random(1), 0)
    b = generate_instance("phishing_lateral", random.Random(1), 1)
    assert a.network_model is not b.network_model


def test_generate_sample_is_reproducible():
    s1 = generate_sample(n_attack_per_archetype=2, n_benign=1, seed=99)
    s2 = generate_sample(n_attack_per_archetype=2, n_benign=1, seed=99)
    assert [s.key for s in s1] == [s.key for s in s2]


# --- viability: the guaranteed path actually works (no LLM) ----------------

def test_phishing_instance_path_is_viable():
    sc = generate_instance("phishing_lateral", random.Random(42), 0)
    sim = Simulation(sc)
    tools = {t.name: t for t in make_red_tools(sim, sc)}
    entry = sc.config["entry_user"]
    target = sc.config["target"]

    out = tools["send_phishing_email"].invoke({"target_user": entry, "pretext": "x"})
    assert "SUCCESS" in out
    out = tools["lateral_movement"].invoke(
        {"from_host": f"ws-{entry}", "to_host": target, "technique": "pass-the-hash"}
    )
    assert "SUCCESS" in out
    out = tools["exfiltrate_data"].invoke({"host": target, "dest_ip": "203.0.113.77"})
    assert "SUCCESS" in out


def test_brute_instance_path_is_viable():
    sc = generate_instance("brute_escalation", random.Random(7), 0)
    sim = Simulation(sc)
    tools = {t.name: t for t in make_red_tools(sim, sc)}
    foothold = sc.config["foothold"]
    weakness = sc.config["weakness"]

    if weakness in ("exploit", "both"):
        recon = json.loads(tools["recon_scan"].invoke({"target": foothold}))
        cves = [v for s in recon["services"] for v in s["vulns"]]
        assert cves, "exploit-path instance must carry a CVE on the foothold"
        out = tools["exploit_known_vulnerability"].invoke(
            {"host": foothold, "cve": cves[0]}
        )
        assert "SUCCESS" in out
    else:
        weak = [
            u for u, a in sim.net.accounts.items()
            if a["password"] == "weak" and not a["mfa"]
        ]
        assert weak, "brute-path instance must have a weak no-MFA account"
        out = tools["brute_force_login"].invoke(
            {"host": foothold, "username": weak[0], "attempts": 8}
        )
        assert "SUCCESS" in out

    out = tools["escalate_privileges"].invoke({"host": foothold})
    assert "SUCCESS" in out
    out = tools["exfiltrate_data"].invoke({"host": foothold, "dest_ip": "198.51.100.99"})
    assert "SUCCESS" in out


# --- statistics -------------------------------------------------------------

def test_wilson_ci_bounds():
    ci = wilson_ci(50, 100)
    assert ci["point"] == 0.5
    assert 0.0 <= ci["low"] <= ci["point"] <= ci["high"] <= 1.0

    ci0 = wilson_ci(0, 10)
    assert ci0["low"] == 0.0 and ci0["high"] > 0.0

    assert wilson_ci(0, 0)["n"] == 0


def test_mean_ci():
    ci = mean_ci([0.2, 0.4, 0.6])
    assert abs(ci["point"] - 0.4) < 1e-9
    assert ci["low"] < ci["point"] < ci["high"]
    assert mean_ci([])["n"] == 0
    single = mean_ci([0.7])
    assert single["low"] == single["high"] == 0.7


def _mk(attack, detection, gt=None, flagged=None, n_logs=100):
    return RunResult(
        scenario_key="s", scenario_name="S", is_attack=attack, detection=detection,
        triage_confidence=0.9, gt_ids=gt or [], gt_by_type={},
        flagged_ids=flagged or [], n_logs=n_logs,
    )


def test_summarize_results_structure():
    results = [
        _mk(True, True, gt=["L1", "L2"], flagged=["L1", "L2"]),
        _mk(True, True, gt=["L1", "L2", "L3"], flagged=["L1"]),
        _mk(False, False, flagged=["L9"], n_logs=100),
    ]
    s = summarize_results(results)
    assert s["n_total"] == 3 and s["n_attack"] == 2 and s["n_benign"] == 1
    assert s["detection"]["tp"] == 2 and s["detection"]["tn"] == 1
    assert 0.0 <= s["detection"]["accuracy"]["point"] <= 1.0
    assert s["evidence"]["recall"]["point"] > 0.0
    assert s["benign_log_fp_rate"]["point"] == 0.01
