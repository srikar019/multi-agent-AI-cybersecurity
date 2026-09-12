"""Interleaved red/blue duel tests (no LLM required).

The red side is a scripted fake agent that invokes the REAL red tools, and the
defender is a scripted function applying the REAL defensive actions -- so the
duel mechanics and the underlying state model are tested end-to-end.
"""
from cyberarena.blue_team.defender import run_defender_turn
from cyberarena.blue_team.defenses import block_ip, reset_credentials
from cyberarena.duel import run_duel, run_duel_eval, summarize_duels
from cyberarena.llm_factory import MockChatModel
from cyberarena.red_team.tools import make_red_tools
from cyberarena.sim.runtime import Simulation
from cyberarena.sim.scenarios import get_scenario

PHISH_EXFIL = [
    ("send_phishing_email", {"target_user": "alice", "pretext": "Invoice"}),
    ("lateral_movement", {"from_host": "ws-alice", "to_host": "db01", "technique": "pass-the-ticket"}),
    ("exfiltrate_data", {"host": "db01", "dest_ip": "203.0.113.77"}),
    ("mark_task_complete", {"summary": "exfiltrated customer db"}),
]


class ScriptedRed:
    """Fake red agent: invokes real tools on a fixed schedule."""

    def __init__(self, sim, steps):
        self.tools = {t.name: t for t in make_red_tools(sim, sim.scenario)}
        self.steps = steps

    def stream(self, inputs, config=None, stream_mode=None):
        for name, args in self.steps:
            self.tools[name].invoke(args)
            yield {"tools": {"messages": []}}


def red_factory(steps):
    def factory(sim):
        return ScriptedRed(sim, steps)

    return factory


def test_duel_red_wins_without_defense():
    res = run_duel(
        get_scenario("phishing_lateral"),
        red_agent=red_factory(PHISH_EXFIL),
        defender_fn=lambda sim: [],
    )
    assert res.red_success is True
    assert res.red_done is True
    assert res.exfiltrated_hosts == ["db01"]
    assert res.red_moves == 4
    # Defender is invoked after each log-producing batch, except the final one
    # (mark_task_complete ends the duel before a defender turn).
    assert res.defender_turns == 3
    assert res.defenses == []


def test_duel_defender_stops_chain_with_block_and_reset():
    def defender(sim):
        attacker = sim.scenario.config.get("attacker_ip", sim.net.attacker_ip)
        applied = [block_ip(sim, attacker)]
        for user in sorted(sim.state["creds"]):
            applied.append(reset_credentials(sim, user))
        # Only actually-applied actions count (mirrors the real defender,
        # which measures actions from the defensive_action log entries).
        return [{"ts": sim.ts, "detail": a} for a in applied if a.startswith("DEFENSE")]

    res = run_duel(
        get_scenario("phishing_lateral"),
        red_agent=red_factory(PHISH_EXFIL),
        defender_fn=defender,
    )
    # Phishing succeeded before the defender acted, but the credential reset
    # killed lateral movement and exfiltration.
    assert res.red_success is False
    assert res.exfiltrated_hosts == []
    assert res.red_done is True  # red CLAIMS completion; success says otherwise
    assert res.blocked_ips == ["198.51.100.23"]
    assert len(res.defenses) == 2
    assert res.defender_turns == 1  # later failed moves produce no new logs


def test_duel_perimeter_block_alone_leaves_foothold_intact():
    def defender(sim):
        attacker = sim.scenario.config.get("attacker_ip", sim.net.attacker_ip)
        out = block_ip(sim, attacker)
        if not out.startswith("DEFENSE"):
            return []
        return [{"ts": sim.ts, "detail": out}]

    res = run_duel(
        get_scenario("phishing_lateral"),
        red_agent=red_factory(PHISH_EXFIL),
        defender_fn=defender,
    )
    # A perimeter block after initial access does NOT stop lateral movement
    # with live harvested credentials -- the defender must also contain those.
    assert res.red_success is True
    assert res.exfiltrated_hosts == ["db01"]


def test_duel_blocking_exfil_destination_prevents_exfil():
    def defender(sim):
        out = block_ip(sim, "203.0.113.77")
        if not out.startswith("DEFENSE"):
            return []
        return [{"ts": sim.ts, "detail": out}]

    res = run_duel(
        get_scenario("phishing_lateral"),
        red_agent=red_factory(PHISH_EXFIL),
        defender_fn=defender,
    )
    assert res.red_success is False
    assert res.red_done is True
    assert res.exfil_attempts_dropped == 1
    assert "203.0.113.77" in res.blocked_ips


def test_duel_defender_every_n_moves():
    calls = {"n": 0}

    def defender(sim):
        calls["n"] += 1
        return []

    res = run_duel(
        get_scenario("phishing_lateral"),
        red_agent=red_factory(PHISH_EXFIL),
        defender_fn=defender,
        defender_every=2,
    )
    # Log-producing batches are moves 1, 2, 3; only move 2 hits the cadence.
    assert calls["n"] == 1
    assert res.defender_turns == 1


def test_defender_turn_with_mock_llm_applies_no_actions():
    sim = Simulation(get_scenario("phishing_lateral"))
    turn = run_defender_turn(MockChatModel(), sim)
    assert turn["actions"] == []
    assert turn["error"] is None
    assert not sim.state["blocked_ips"]
    assert not sim.state["isolated_hosts"]


def test_duel_with_mock_llm_smoke():
    res = run_duel(get_scenario("phishing_lateral"), llm=MockChatModel())
    assert res.red_success is False
    assert res.red_moves == 0
    assert res.defender_turns == 0
    assert res.n_logs > 0


# --- Phase C: batch duel evaluation + statistics ----------------------------


def _block_and_reset_defender(sim):
    attacker = sim.scenario.config.get("attacker_ip", sim.net.attacker_ip)
    applied = [block_ip(sim, attacker)]
    for user in sorted(sim.state["creds"]):
        applied.append(reset_credentials(sim, user))
    return [
        {
            "ts": sim.ts,
            "detail": a,
            "action": "block_ip" if "blocked IP" in a else "reset_credentials",
        }
        for a in applied
        if a.startswith("DEFENSE")
    ]


def test_summarize_duels_counts_and_rates():
    win = run_duel(
        get_scenario("phishing_lateral"),
        red_agent=red_factory(PHISH_EXFIL),
        defender_fn=lambda sim: [],
    )
    loss = run_duel(
        get_scenario("phishing_lateral"),
        red_agent=red_factory(PHISH_EXFIL),
        defender_fn=_block_and_reset_defender,
    )
    s = summarize_duels([win, loss])
    assert s["n_duels"] == 2
    assert s["red_wins"] == 1
    assert s["red_win_rate"]["point"] == 0.5
    assert s["red_claim_rate"]["point"] == 1.0  # both reds claimed completion
    assert s["defense_actions"] == {"block_ip": 1, "reset_credentials": 1}
    assert s["defense_actions_total"] == 2


def test_duel_eval_fixed_scripted_offline(tmp_path):
    results, summary = run_duel_eval(
        fixed=True,
        llm=MockChatModel(),
        out_dir=str(tmp_path),
        red_agent=red_factory(PHISH_EXFIL),
        defender_fn=lambda sim: [],
    )
    # Two fixed attack scenarios, one duel each; unopposed red wins both.
    assert summary["n_duels"] == 2
    assert summary["red_wins"] == 2
    assert summary["red_win_rate"]["point"] == 1.0
    assert all(r.red_success for r in results)
    assert len(list(tmp_path.glob("duelval_*.json"))) == 1

# --- Red adaptation: recovery from defender actions --------------------------

def test_red_recovers_from_blocked_exfil_destination():
    """Exfil dest blocked -> red retries to a fresh pool IP and still wins."""
    def defender(sim):
        out = block_ip(sim, "203.0.113.77")
        if not out.startswith("DEFENSE"):
            return []
        return [{"ts": sim.ts, "detail": out, "action": "block_ip"}]

    res = run_duel(
        get_scenario("phishing_lateral"),
        red_agent=red_factory([
            ("send_phishing_email", {"target_user": "alice", "pretext": "Invoice"}),
            ("lateral_movement", {"from_host": "ws-alice", "to_host": "db01", "technique": "pass-the-ticket"}),
            ("exfiltrate_data", {"host": "db01", "dest_ip": "203.0.113.77"}),   # dropped
            ("exfiltrate_data", {"host": "db01", "dest_ip": "198.51.100.47"}),  # retry wins
            ("mark_task_complete", {"summary": "exfil via alternate destination"}),
        ]),
        defender_fn=defender,
    )
    assert res.exfil_attempts_dropped == 1
    assert res.red_success is True
    assert res.exfiltrated_hosts == ["db01"]
    assert "198.51.100.47" not in res.blocked_ips


def test_red_recovers_from_source_block_and_cred_reset_via_rotation():
    """Source IP blocked + creds reset -> red rotates, re-phishes another user,
    pivots through a new path, and completes the objective."""
    from cyberarena.blue_team.defenses import reset_credentials

    def defender(sim):
        # One-shot response (mirrors a single LLM defender turn): block the
        # original source IP and reset the credentials harvested so far.
        if getattr(defender, "done", False):
            return []
        defender.done = True
        attacker = sim.scenario.config.get("attacker_ip", sim.net.attacker_ip)
        applied = [block_ip(sim, attacker)]
        applied.extend(reset_credentials(sim, u) for u in sorted(sim.state["creds"]))
        return [
            {"ts": sim.ts, "detail": a}
            for a in applied
            if a.startswith("DEFENSE")
        ]

    res = run_duel(
        get_scenario("phishing_lateral"),
        red_agent=red_factory([
            ("send_phishing_email", {"target_user": "alice", "pretext": "Invoice"}),
            # defender: block source IP + reset alice's creds
            ("rotate_infrastructure", {"reason": "source IP blocked"}),
            ("send_phishing_email", {"target_user": "dave", "pretext": "Payroll"}),  # foothold on mail01
            ("lateral_movement", {"from_host": "mail01", "to_host": "ws-bob", "technique": "pass-the-hash"}),
            ("lateral_movement", {"from_host": "ws-bob", "to_host": "db01", "technique": "pass-the-hash"}),
            ("exfiltrate_data", {"host": "db01", "dest_ip": "203.0.113.77"}),
            ("mark_task_complete", {"summary": "re-entered via dave after rotation"}),
        ]),
        defender_fn=defender,
    )
    assert res.red_success is True
    assert res.exfiltrated_hosts == ["db01"]
    assert res.defenses, "defender actions should have been applied"
    # red adapted: it kept operating from a fresh pool address; success proves
    # the re-phishing of 'dave' happened AFTER the original IP was blocked.
    assert "198.51.100.23" in res.blocked_ips


def test_tool_feedback_tells_red_how_to_adapt():
    """Failure messages must contain actionable recovery guidance."""
    sim = Simulation(get_scenario("phishing_lateral"))
    tools = {t.name: t for t in make_red_tools(sim, sim.scenario)}

    # Blocked exfil destination -> suggests retrying with a different one.
    block_ip(sim, "203.0.113.77")
    tools["send_phishing_email"].invoke({"target_user": "alice", "pretext": "Invoice"})
    tools["lateral_movement"].invoke(
        {"from_host": "ws-alice", "to_host": "db01", "technique": "pass-the-ticket"}
    )
    out = tools["exfiltrate_data"].invoke({"host": "db01", "dest_ip": "203.0.113.77"})
    assert out.startswith("FAILED")
    assert "DIFFERENT external destination" in out
    assert "198.51.100.47" in out  # names a pool address to retry with

    # Rotation -> tells red internal footholds survive a source-IP block.
    out = tools["rotate_infrastructure"].invoke({"reason": "blocked"})
    assert out.startswith("SUCCESS")
    assert "unaffected" in out

    # Isolated host -> suggests re-establishing access another way.
    from cyberarena.blue_team.defenses import isolate_host
    isolate_host(sim, "db01")
    out = tools["exfiltrate_data"].invoke({"host": "db01", "dest_ip": "198.51.100.47"})
    assert "isolated" in out
    assert "rotate_infrastructure" in out
