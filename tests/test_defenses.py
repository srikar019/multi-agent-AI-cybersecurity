"""Blue Team defensive action tests (no LLM required).

Phase A of the red/blue duel: defensive actions must ACTUALLY remove
red-team capability, verified end-to-end through the red team tools.
"""
from cyberarena.blue_team.defenses import block_ip, isolate_host, reset_credentials
from cyberarena.red_team.tools import make_red_tools
from cyberarena.sim.runtime import Simulation
from cyberarena.sim.scenarios import get_scenario


def _setup(key: str = "phishing_lateral"):
    sim = Simulation(get_scenario(key))
    tools = {t.name: t for t in make_red_tools(sim, sim.scenario)}
    attacker = sim.scenario.config.get("attacker_ip", sim.net.attacker_ip)
    return sim, tools, attacker


def test_block_ip_stops_external_recon_phishing_brute_exploit():
    sim, tools, attacker = _setup()
    block_ip(sim, attacker)

    out = tools["recon_scan"].invoke({"target": "mail01"})
    assert "blocked" in out

    out = tools["send_phishing_email"].invoke({"target_user": "alice", "pretext": "hi"})
    assert "blocked" in out.lower()

    out = tools["brute_force_login"].invoke(
        {"host": "web01", "username": "svc-backup", "attempts": 6}
    )
    assert "blocked" in out.lower()

    out = tools["exploit_known_vulnerability"].invoke(
        {"host": "web01", "cve": "CVE-2023-1234"}
    )
    assert "blocked" in out.lower()

    assert not sim.state["creds"]
    assert not sim.state["compromised"]


def test_blocking_unrelated_ip_does_not_stop_attack():
    sim, tools, attacker = _setup()
    block_ip(sim, "203.0.113.5")
    out = tools["send_phishing_email"].invoke({"target_user": "alice", "pretext": "hi"})
    assert "SUCCESS" in out


def test_block_ip_stops_exfil_to_blocked_destination():
    sim, tools, attacker = _setup()
    tools["send_phishing_email"].invoke({"target_user": "alice", "pretext": "Invoice"})
    tools["lateral_movement"].invoke(
        {"from_host": "ws-alice", "to_host": "db01", "technique": "pass-the-ticket"}
    )

    block_ip(sim, "203.0.113.77")
    out = tools["exfiltrate_data"].invoke({"host": "db01", "dest_ip": "203.0.113.77"})
    assert "SUCCESS" not in out
    assert "blocked" in out.lower()

    out = tools["exfiltrate_data"].invoke({"host": "db01", "dest_ip": "203.0.113.99"})
    assert "SUCCESS" in out


def test_isolate_host_removes_foothold_and_blocks_reattack():
    sim, tools, attacker = _setup("bruteforce_escalation")
    out = tools["exploit_known_vulnerability"].invoke(
        {"host": "web01", "cve": "CVE-2023-1234"}
    )
    assert "SUCCESS" in out
    assert "web01" in sim.state["compromised"]

    isolate_host(sim, "web01")
    assert "web01" not in sim.state["compromised"]
    assert "web01" in sim.state["isolated_hosts"]

    out = tools["exploit_known_vulnerability"].invoke(
        {"host": "web01", "cve": "CVE-2023-1234"}
    )
    assert "not reachable" in out
    assert "web01" not in sim.state["compromised"]

    out = tools["escalate_privileges"].invoke({"host": "web01"})
    assert "isolated" in out.lower()


def test_isolate_host_blocks_lateral_target():
    sim, tools, attacker = _setup()
    tools["send_phishing_email"].invoke({"target_user": "alice", "pretext": "Invoice"})
    isolate_host(sim, "db01")
    out = tools["lateral_movement"].invoke(
        {"from_host": "ws-alice", "to_host": "db01", "technique": "pass-the-ticket"}
    )
    assert "not reachable" in out
    assert "db01" not in sim.state["compromised"]


def test_isolate_foothold_blocks_lateral_source_and_exfil():
    sim, tools, attacker = _setup()
    tools["send_phishing_email"].invoke({"target_user": "alice", "pretext": "Invoice"})
    tools["lateral_movement"].invoke(
        {"from_host": "ws-alice", "to_host": "db01", "technique": "pass-the-ticket"}
    )

    isolate_host(sim, "ws-alice")
    out = tools["lateral_movement"].invoke(
        {"from_host": "ws-alice", "to_host": "dc01", "technique": "pass-the-ticket"}
    )
    assert "isolated" in out.lower()

    isolate_host(sim, "db01")
    out = tools["exfiltrate_data"].invoke({"host": "db01", "dest_ip": "203.0.113.77"})
    assert "isolated" in out.lower()


def test_reset_credentials_blocks_lateral():
    sim, tools, attacker = _setup()
    tools["send_phishing_email"].invoke({"target_user": "alice", "pretext": "Invoice"})
    assert sim.state["creds"].get("alice")

    reset_credentials(sim, "alice")
    assert "alice" not in sim.state["creds"]

    out = tools["lateral_movement"].invoke(
        {"from_host": "ws-alice", "to_host": "db01", "technique": "pass-the-ticket"}
    )
    assert "need harvested credentials" in out
    assert "db01" not in sim.state["compromised"]


def test_defensive_actions_logged_without_ground_truth():
    sim, tools, attacker = _setup()
    tools["send_phishing_email"].invoke({"target_user": "alice", "pretext": "Invoice"})
    block_ip(sim, attacker)
    isolate_host(sim, "db01")
    reset_credentials(sim, "alice")

    defense_logs = [e for e in sim.logs.snapshot() if e["event"] == "defensive_action"]
    assert len(defense_logs) == 3
    assert all(e["source"] == "soc" for e in defense_logs)
    defense_ids = {e["id"] for e in defense_logs}
    assert not (defense_ids & set(sim.attack_log_ids()))


def test_defense_idempotence_and_unknown_host():
    sim, tools, attacker = _setup()
    block_ip(sim, attacker)
    out = block_ip(sim, attacker)
    assert "already" in out

    out = isolate_host(sim, "nosuch01")
    assert "unknown host" in out
    out = isolate_host(sim, "db01")
    assert "DEFENSE" in out
    out = isolate_host(sim, "db01")
    assert "already" in out

    out = reset_credentials(sim, "bob")
    assert "No active harvested credentials" in out


def test_early_block_prevents_full_attack_chain():
    sim, tools, attacker = _setup()
    block_ip(sim, attacker)
    for call in (
        ("send_phishing_email", {"target_user": "alice", "pretext": "hi"}),
        ("brute_force_login", {"host": "web01", "username": "svc-backup", "attempts": 8}),
        ("exploit_known_vulnerability", {"host": "web01", "cve": "CVE-2023-1234"}),
    ):
        out = tools[call[0]].invoke(call[1])
        assert "SUCCESS" not in out

    assert not sim.state["compromised"]
    assert not sim.state["creds"]
    by_type = sim.attack_ids_by_type()
    assert "lateral_movement" not in by_type
    assert "exfiltration" not in by_type


# --- red infrastructure rotation --------------------------------------------


def test_rotate_infrastructure_unblocks_attack():
    sim, tools, attacker = _setup()
    block_ip(sim, attacker)
    out = tools["send_phishing_email"].invoke({"target_user": "alice", "pretext": "hi"})
    assert "blocked" in out.lower()

    out = tools["rotate_infrastructure"].invoke({"reason": "source IP blocked"})
    assert "SUCCESS" in out
    new_ip = sim.state["attacker_ip"]
    assert new_ip != attacker
    assert new_ip not in sim.state["blocked_ips"]

    out = tools["send_phishing_email"].invoke({"target_user": "alice", "pretext": "hi"})
    assert "SUCCESS" in out
    assert sim.state["creds"].get("alice") == new_ip


def test_rotate_infrastructure_skips_blocked_addresses():
    sim, tools, attacker = _setup()
    pool = list(sim.net.attacker_pool)
    block_ip(sim, pool[0])
    block_ip(sim, pool[1])
    tools["rotate_infrastructure"].invoke({"reason": "blocked"})
    assert sim.state["attacker_ip"] == pool[2]


def test_rotate_infrastructure_exhausted():
    sim, tools, attacker = _setup()
    for ip in sim.net.attacker_pool:
        block_ip(sim, ip)
    out = tools["rotate_infrastructure"].invoke({"reason": "blocked"})
    assert "FAILED" in out
    assert sim.state["attacker_ip"] == attacker


def test_rotation_leaves_no_log_trace():
    sim, tools, attacker = _setup()
    before = sim.total_logs()
    tools["rotate_infrastructure"].invoke({"reason": "opsec"})
    assert sim.total_logs() == before
