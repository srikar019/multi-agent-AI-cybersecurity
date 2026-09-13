"""MCP P1 tests: sessions, role separation, resources, agent-via-MCP wiring."""
import json

from cyberarena import mcp_client, mcp_server
from cyberarena.llm_factory import get_chat_model


def _sid(payload: str) -> str:
    return json.loads(payload)["session_id"]


def test_legacy_default_session_still_works():
    out = mcp_server.reset_scenario(scenario_key="phishing_lateral")
    assert "ready" in out
    assert "mail01" in mcp_server.get_network_state()
    assert "mail01" in mcp_server.recon_scan(target="mail01")


def test_session_isolation():
    a = _sid(mcp_server.create_session(scenario_key="phishing_lateral", role="duel"))
    b = _sid(mcp_server.create_session(scenario_key="bruteforce_escalation", role="duel"))
    assert a != b
    mcp_server.recon_scan(session_id=a, target="mail01")
    na = mcp_server.get_network_state(session_id=a)
    nb = mcp_server.get_network_state(session_id=b)
    assert na != nb  # different sims, different log counts


def test_role_separation():
    red = _sid(mcp_server.create_session(scenario_key="phishing_lateral", role="red"))
    blue = _sid(mcp_server.create_session(scenario_key="phishing_lateral", role="blue"))
    # blue must not run attack tools
    denied = mcp_server.exfiltrate_data(session_id=blue, host="db01", dest_ip="1.1.1.1")
    assert "DENIED" in denied
    # red must not run defense tools
    denied2 = mcp_server.block_ip(session_id=red, ip="1.2.3.4")
    assert "DENIED" in denied2
    # allowed paths still work
    assert "mail01" in mcp_server.recon_scan(session_id=red, target="mail01")
    assert isinstance(mcp_server.search_logs(session_id=blue, keyword="mail"), str)


def test_resources():
    sid = _sid(mcp_server.create_session(scenario_key="phishing_lateral", role="duel"))
    mcp_server.recon_scan(session_id=sid, target="mail01")
    assert "mail01" in mcp_server.res_network_state(sid)
    assert isinstance(mcp_server.res_logs_current(sid), str)
    assert isinstance(mcp_server.res_incident_timeline(sid), str)
    sc = json.loads(mcp_server.res_scenario_current(sid))
    assert sc["scenario_key"] == "phishing_lateral"


def test_agents_operate_via_mcp_mock_llm():
    llm = get_chat_model("mock")
    sid = _sid(mcp_server.create_session(scenario_key="phishing_lateral", role="duel"))
    red_agent = mcp_client.build_red_agent_via_mcp(llm, sid, max_steps=2)
    analyst = mcp_client.build_analyst_agent_via_mcp(llm, sid)
    defender = mcp_client.build_defender_agent_via_mcp(llm, sid)
    # tool names prove agents go through the MCP layer
    red_names = {t.name for t in mcp_client.make_mcp_red_tools(sid)}
    assert {"recon_scan", "exfiltrate_data"} <= red_names
    # mock LLM smoke: analyst + defender invoke without crashing
    analyst.invoke({"messages": [{"role": "user", "content": "Investigate."}]},
                   config={"recursion_limit": 10})
    defender.invoke({"messages": [{"role": "user", "content": "PASS if benign."}]},
                    config={"recursion_limit": 10})
