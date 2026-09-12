"""CyberArena MCP Server: standardized tool interface over the simulation.

This is an *adapter layer* on top of the existing in-process tools — it does
not replace them. Eval/tests keep using the fast in-process path:

    make_red_tools(sim, scenario)      in red_team/tools.py
    make_analyst_tools(view)           in blue_team/agents.py
    block_ip / isolate_host / ...      in blue_team/defenses.py

The MCP server holds one live ``Simulation`` and delegates every MCP tool
call to those same factories, so logic stays in one place.

Run:
    python -m cyberarena.mcp_server        # stdio transport (default)
    python -m cyberarena.mcp_server --help # other transports

Any MCP client (Claude Desktop, VS Code, langchain-mcp-adapters, etc.)
can then drive Red or Blue against the same tool ecosystem, with any LLM.
"""
from __future__ import annotations

import argparse
import json

from mcp.server.fastmcp import FastMCP

from .blue_team import defenses
from .blue_team.agents import make_analyst_tools
from .blue_team.log_view import LogView
from .red_team.tools import make_red_tools
from .sim.runtime import Simulation
from .sim.scenarios import all_scenario_keys, get_scenario

mcp = FastMCP("cyberarena")

# Single live simulation. Reset via reset_scenario().
_CTX: dict = {"sim": None, "scenario": None}


def _ensure_sim() -> Simulation:
    sim = _CTX.get("sim")
    if sim is None:
        reset_scenario("phishing_lateral")
        sim = _CTX["sim"]
    return sim


def _red_tool(name: str, args: dict) -> str:
    sim = _ensure_sim()
    scenario = _CTX["scenario"]
    for t in make_red_tools(sim, scenario):
        if t.name == name:
            return str(t.invoke(args))
    return f"unknown red tool '{name}'"


def _analyst_tool(name: str, args: dict) -> str:
    sim = _ensure_sim()
    view = LogView(sim.logs.snapshot())
    for t in make_analyst_tools(view):
        if t.name == name:
            return str(t.invoke(args))
    return f"unknown analyst tool '{name}'"


# --- scenario / state -------------------------------------------------------

@mcp.tool()
def reset_scenario(scenario_key: str = "phishing_lateral") -> str:
    """Start a fresh simulation for a scenario. Must be called first."""
    scenario = get_scenario(scenario_key)
    _CTX["scenario"] = scenario
    _CTX["sim"] = Simulation(scenario)
    sim = _CTX["sim"]
    return (
        f"scenario '{scenario_key}' ready: {scenario.name}\n"
        f"{sim.net.network_summary()}\n"
        f"Objective: {scenario.objective}"
    )


@mcp.tool()
def get_network_state() -> str:
    """Topology overview + live defense/compromise state (no versions/CVEs)."""
    sim = _ensure_sim()
    state = sim.state
    return (
        f"{sim.net.network_summary()}\n\n"
        f"live state: compromised={sorted(state['compromised'])} "
        f"isolated={sorted(state['isolated_hosts'])} "
        f"blocked_ips={sorted(state['blocked_ips'])} "
        f"exfiltrated={sorted(state['exfiltrated'])} "
        f"total_logs={len(sim.logs)}"
    )


@mcp.tool()
def get_digest() -> str:
    """Log statistics + notable events digest (Blue triage input)."""
    return _ensure_sim().digest()


@mcp.tool()
def list_scenarios() -> str:
    """List available scenario keys."""
    return json.dumps(all_scenario_keys())


# --- Red Team (delegates to red_team/tools.py) -------------------------------

@mcp.tool()
def recon_scan(target: str = "") -> str:
    """Enumerate a host: services, versions, CVEs, accounts, reachability."""
    return _red_tool("recon_scan", {"target": target})


@mcp.tool()
def send_phishing_email(target_user: str, pretext: str) -> str:
    """Send a spear-phishing email to a user."""
    return _red_tool("send_phishing_email", {"target_user": target_user, "pretext": pretext})


@mcp.tool()
def brute_force_login(host: str, username: str, attempts: int) -> str:
    """Password-guessing against a host login service."""
    return _red_tool("brute_force_login", {"host": host, "username": username, "attempts": attempts})


@mcp.tool()
def exploit_known_vulnerability(host: str, cve: str) -> str:
    """Attempt to exploit a CVE on a host."""
    return _red_tool("exploit_known_vulnerability", {"host": host, "cve": cve})


@mcp.tool()
def escalate_privileges(host: str) -> str:
    """Escalate to root/admin on a compromised host."""
    return _red_tool("escalate_privileges", {"host": host})


@mcp.tool()
def lateral_movement(from_host: str, to_host: str, technique: str) -> str:
    """Move laterally using harvested creds or admin access."""
    return _red_tool(
        "lateral_movement",
        {"from_host": from_host, "to_host": to_host, "technique": technique},
    )


@mcp.tool()
def exfiltrate_data(host: str, dest_ip: str) -> str:
    """Exfiltrate data from a compromised host to an external IP."""
    return _red_tool("exfiltrate_data", {"host": host, "dest_ip": dest_ip})


@mcp.tool()
def rotate_infrastructure(reason: str) -> str:
    """Switch active external source IP after a perimeter block."""
    return _red_tool("rotate_infrastructure", {"reason": reason})


@mcp.tool()
def mark_task_complete(summary: str) -> str:
    """Mark the operation objective as achieved."""
    return _red_tool("mark_task_complete", {"summary": summary})


# --- Blue Team read-only (delegates to blue_team/agents.py) ------------------

@mcp.tool()
def search_logs(keyword: str) -> str:
    """Full-text search across sanitized log lines (alias: query_logs)."""
    return _analyst_tool("search_logs", {"keyword": keyword})


@mcp.tool()
def query_logs(keyword: str) -> str:
    """Alias of search_logs for MCP clients expecting query_logs."""
    return _analyst_tool("search_logs", {"keyword": keyword})


@mcp.tool()
def filter_logs(event_types: list[str] | None = None, hosts: list[str] | None = None, src_ips: list[str] | None = None) -> str:
    """Filter logs by event types, hosts, and/or source IPs."""
    return _analyst_tool(
        "filter_logs",
        {"event_types": event_types or [], "hosts": hosts or [], "src_ips": src_ips or []},
    )


@mcp.tool()
def get_log_line(log_id: str) -> str:
    """Fetch a single sanitized log line by ID (e.g. L0042)."""
    return _analyst_tool("get_log_line", {"log_id": log_id})


@mcp.tool()
def get_log_stats(limit: int = 20) -> str:
    """Most common event types, hosts, and source IPs."""
    return _analyst_tool("get_log_stats", {"limit": limit})


@mcp.tool()
def inspect_host(host: str) -> str:
    """Recent log lines for one host (convenience over filter_logs)."""
    return _analyst_tool("filter_logs", {"event_types": [], "hosts": [host], "src_ips": []})


# --- Blue Team defenses (delegates to blue_team/defenses.py) -----------------

@mcp.tool()
def block_ip(ip: str) -> str:
    """Perimeter-block an IP (inbound source + outbound destination)."""
    return str(defenses.block_ip(_ensure_sim(), ip))


@mcp.tool()
def isolate_host(host: str) -> str:
    """Isolate a host; revokes any attacker foothold on it."""
    return str(defenses.isolate_host(_ensure_sim(), host))


@mcp.tool()
def reset_credentials(username: str) -> str:
    """Reset a user's credentials, invalidating harvested sessions."""
    return str(defenses.reset_credentials(_ensure_sim(), username))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cyberarena-mcp", description="CyberArena MCP server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
