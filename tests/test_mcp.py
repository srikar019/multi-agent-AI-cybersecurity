"""MCP server adapter tests (no LLM, no transport required)."""
from cyberarena import mcp_server


def test_mcp_reset_and_red_blue_tools():
    out = mcp_server.reset_scenario(scenario_key="phishing_lateral")
    assert "ready" in out

    net = mcp_server.get_network_state()
    assert "mail01" in net

    recon = mcp_server.recon_scan(target="mail01")
    assert "mail01" in recon

    search = mcp_server.search_logs(keyword="mail")
    assert isinstance(search, str)

    stats = mcp_server.get_log_stats(limit=5)
    assert "total lines" in stats

    blocked = mcp_server.block_ip(ip="198.51.100.23")
    assert "blocked" in blocked.lower()

    state = mcp_server.get_network_state()
    assert "198.51.100.23" in state
