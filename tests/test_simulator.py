"""Simulator + Red Team tool tests (no LLM required)."""
from cyberarena.red_team.tools import make_red_tools
from cyberarena.sim.digest import build_digest
from cyberarena.sim.log_store import render_line
from cyberarena.sim.network import network_summary
from cyberarena.sim.runtime import Simulation
from cyberarena.sim.scenarios import get_scenario


def test_network_summary_lists_hosts():
    text = network_summary()
    assert "web01" in text and "db01" in text and "198.51.100.23" in text


def test_benign_simulation_has_no_ground_truth():
    sim = Simulation(get_scenario("benign_baseline"))
    assert sim.total_logs() > 50
    assert sim.attack_log_ids() == []


def test_snapshot_strips_ground_truth():
    sim = Simulation(get_scenario("benign_baseline"))
    sim.tick(1)
    sim.add(
        source="auth", host="ws-alice", user="alice", src_ip="10.0.1.20",
        dst_ip="10.0.1.20", event="login_success", detail="test", gt="initial_access",
    )
    snap = sim.logs.snapshot()
    assert all("_meta" not in e for e in snap)
    assert sim.attack_log_ids()
    assert snap[-1]["id"] in sim.attack_log_ids()


def test_red_tools_build_attack_chain():
    sim = Simulation(get_scenario("phishing_lateral"))
    tools = {t.name: t for t in make_red_tools(sim, sim.scenario)}

    out = tools["recon_scan"].invoke({"target": "mail01"})
    assert "mail01" in out

    out = tools["send_phishing_email"].invoke(
        {"target_user": "alice", "pretext": "Invoice due"}
    )
    assert "SUCCESS" in out

    out = tools["lateral_movement"].invoke(
        {"from_host": "ws-alice", "to_host": "db01", "technique": "pass-the-ticket"}
    )
    assert "SUCCESS" in out

    out = tools["exfiltrate_data"].invoke(
        {"host": "db01", "dest_ip": "203.0.113.77"}
    )
    assert "SUCCESS" in out

    tools["mark_task_complete"].invoke({"summary": "exfiltrated customer db"})

    by_type = sim.attack_ids_by_type()
    assert {"recon", "phishing", "initial_access", "lateral_movement", "exfiltration"} <= set(by_type)
    assert sim.state["done"] is True
    assert sim.state["creds"].get("alice")


def test_red_tools_require_foothold_before_lateral():
    sim = Simulation(get_scenario("phishing_lateral"))
    tools = {t.name: t for t in make_red_tools(sim, sim.scenario)}
    out = tools["lateral_movement"].invoke(
        {"from_host": "ws-alice", "to_host": "db01", "technique": "pass-the-ticket"}
    )
    assert "need harvested credentials" in out


def test_brute_force_threshold():
    sim = Simulation(get_scenario("bruteforce_escalation"))
    tools = {t.name: t for t in make_red_tools(sim, sim.scenario)}
    out = tools["brute_force_login"].invoke(
        {"host": "web01", "username": "svc-backup", "attempts": 6}
    )
    assert "SUCCESS" in out
    assert sim.state["creds"].get("svc-backup")

    out = tools["brute_force_login"].invoke(
        {"host": "web01", "username": "admin", "attempts": 3}
    )
    assert "Failed" in out


def test_digest_contains_counts_and_notables():
    sim = Simulation(get_scenario("benign_baseline"))
    d = build_digest(sim.logs.snapshot())
    assert "TOTAL LOG LINES" in d
    assert "auth/login_failure" in d


def test_render_line_contains_id():
    sim = Simulation(get_scenario("benign_baseline"))
    snap = sim.logs.snapshot()
    line = render_line(snap[0])
    assert snap[0]["id"] in line


# --- emergent state-model tests (no answer key) ---------------------------

def test_recon_reveals_versions_and_vulns():
    sim = Simulation(get_scenario("bruteforce_escalation"))
    tools = {t.name: t for t in make_red_tools(sim, sim.scenario)}
    out = tools["recon_scan"].invoke({"target": "web01"})
    assert "CVE-2023-1234" in out
    assert '"exposed": true' in out


def test_network_summary_does_not_leak_vulns():
    text = network_summary()
    assert "CVE-2023-1234" not in text
    assert "CVE-2021-44026" not in text


def test_phishing_trained_user_fails():
    sim = Simulation(get_scenario("phishing_lateral"))
    tools = {t.name: t for t in make_red_tools(sim, sim.scenario)}
    out = tools["send_phishing_email"].invoke({"target_user": "bob", "pretext": "hi"})
    assert "SUCCESS" not in out
    assert "bob" not in sim.state["creds"]


def test_exploit_requires_matching_cve():
    sim = Simulation(get_scenario("bruteforce_escalation"))
    tools = {t.name: t for t in make_red_tools(sim, sim.scenario)}
    out = tools["exploit_known_vulnerability"].invoke(
        {"host": "web01", "cve": "CVE-9999-9999"}
    )
    assert "failed" in out.lower()
    out = tools["exploit_known_vulnerability"].invoke(
        {"host": "web01", "cve": "CVE-2023-1234"}
    )
    assert "SUCCESS" in out
    assert "web01" in sim.state["compromised"]


def test_exploit_blocked_when_unreachable():
    sim = Simulation(get_scenario("phishing_lateral"))
    tools = {t.name: t for t in make_red_tools(sim, sim.scenario)}
    # db01 is internal and there is no foothold yet -> not reachable from external
    out = tools["exploit_known_vulnerability"].invoke(
        {"host": "db01", "cve": "CVE-2023-1234"}
    )
    assert "not reachable" in out


def test_lateral_blocked_by_segmentation():
    sim = Simulation(get_scenario("bruteforce_escalation"))
    tools = {t.name: t for t in make_red_tools(sim, sim.scenario)}
    tools["brute_force_login"].invoke(
        {"host": "web01", "username": "svc-backup", "attempts": 6}
    )
    # web01 can reach db01/dc01 but NOT ws-alice
    out = tools["lateral_movement"].invoke(
        {"from_host": "web01", "to_host": "ws-alice", "technique": "psexec"}
    )
    assert "not reachable" in out


def test_escalate_requires_privesc_vector():
    sim = Simulation(get_scenario("bruteforce_escalation"))
    tools = {t.name: t for t in make_red_tools(sim, sim.scenario)}
    tools["exploit_known_vulnerability"].invoke(
        {"host": "web01", "cve": "CVE-2023-1234"}
    )
    out = tools["escalate_privileges"].invoke({"host": "web01"})
    assert "SUCCESS" in out
    # mail01 has no local privilege-escalation vector
    tools["exploit_known_vulnerability"].invoke(
        {"host": "mail01", "cve": "CVE-2021-44026"}
    )
    out = tools["escalate_privileges"].invoke({"host": "mail01"})
    assert "SUCCESS" not in out


def test_brute_force_blocked_by_mfa():
    sim = Simulation(get_scenario("phishing_lateral"))
    tools = {t.name: t for t in make_red_tools(sim, sim.scenario)}
    # bob has a strong password AND MFA -> brute force fails even with attempts
    out = tools["brute_force_login"].invoke(
        {"host": "web01", "username": "bob", "attempts": 20}
    )
    assert "Failed" in out
    assert "bob" not in sim.state["creds"]