"""CyberArena MCP Server: session-aware, role-separated tool/resource layer.

Design (P1):
  - Sessions, not a global sim: ``create_session(scenario_key, role)``
    returns a ``session_id``. Each session owns one ``Simulation``.
    ``sessions = {session_id: {"sim", "scenario", "role"}}``.
  - Role separation: ``red`` sessions get attack tools only,
    ``blue`` sessions get log/SIEM + defense tools only,
    ``duel`` sessions (orchestrator) get both. Wrong-role calls are
    rejected with an explicit error string.
  - Resources (read-only): network state, logs, incident timeline,
    scenario config — ``Tool = act``, ``Resource = inspect``.
  - Delegation: every tool still delegates to the existing factories
    (``make_red_tools`` / ``make_analyst_tools`` / ``defenses``) so the
    eval/tests fast path is untouched.

Legacy no-session calls (``reset_scenario()`` etc. without ``session_id``)
keep working against a ``"default"`` duel session for backward compat.
"""
from __future__ import annotations

import argparse
import json
import uuid

from mcp.server.fastmcp import FastMCP

from .blue_team import defenses
from .blue_team.agents import make_analyst_tools
from .blue_team.log_view import LogView
from .red_team.tools import make_red_tools
from .sim.log_store import render_line
from .sim.runtime import Simulation
from .sim.scenarios import all_scenario_keys, get_scenario

mcp = FastMCP("cyberarena")

ROLES = ("red", "blue", "duel")

RED_TOOLS = {
    "recon_scan", "send_phishing_email", "brute_force_login",
    "exploit_known_vulnerability", "escalate_privileges",
    "lateral_movement", "exfiltrate_data", "rotate_infrastructure",
    "mark_task_complete",
}
BLUE_READ_TOOLS = {
    "search_logs", "query_logs", "filter_logs",
    "get_log_line", "get_log_stats", "inspect_host",
    "get_digest", "get_network_state",
}
BLUE_DEFENSE_TOOLS = {"block_ip", "isolate_host", "reset_credentials"}

ROLE_ALLOW = {
    "red": RED_TOOLS | {"get_network_state", "list_scenarios"},
    "blue": BLUE_READ_TOOLS | BLUE_DEFENSE_TOOLS | {"list_scenarios"},
    "duel": RED_TOOLS | BLUE_READ_TOOLS | BLUE_DEFENSE_TOOLS | {"list_scenarios", "get_digest", "get_network_state"},
}

sessions: dict[str, dict] = {}
_DEFAULT = "default"


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _resolve(session_id: str | None) -> dict:
    sid = session_id or _DEFAULT
    if sid not in sessions:
        if sid == _DEFAULT:
            create_session("phishing_lateral", role="duel", session_id=_DEFAULT)
        else:
            raise ValueError(f"unknown session '{sid}'. Call create_session first.")
    return sessions[sid]


def _check(session_id: str | None, tool: str) -> tuple[dict, str | None]:
    try:
        entry = _resolve(session_id)
    except ValueError as exc:
        return {}, str(exc)
    if tool not in ROLE_ALLOW.get(entry["role"], set()):
        return {}, (
            f"DENIED: tool '{tool}' not allowed for role '{entry['role']}'. "
            f"session_id={session_id or _DEFAULT}"
        )
    return entry, None


def _red_tool(entry: dict, name: str, args: dict) -> str:
    for t in make_red_tools(entry["sim"], entry["scenario"]):
        if t.name == name:
            return str(t.invoke(args))
    return f"unknown red tool '{name}'"


def _analyst_tool(entry: dict, name: str, args: dict) -> str:
    view = LogView(entry["sim"].logs.snapshot())
    for t in make_analyst_tools(view):
        if t.name == name:
            return str(t.invoke(args))
    return f"unknown analyst tool '{name}'"


# --- sessions ---------------------------------------------------------------

@mcp.tool()
def create_session(scenario_key: str = "phishing_lateral", role: str = "duel", session_id: str | None = None) -> str:
    """Create an isolated simulation session. Returns a session_id for all later calls."""
    role = (role or "duel").lower()
    if role not in ROLES:
        return f"unknown role '{role}'. Choose from {list(ROLES)}"
    scenario = get_scenario(scenario_key)
    sid = session_id or _new_id()
    sessions[sid] = {"sim": scenario and Simulation(scenario), "scenario": scenario, "role": role}
    return json.dumps({"session_id": sid, "scenario": scenario_key, "role": role})


@mcp.tool()
def reset_scenario(scenario_key: str = "phishing_lateral", session_id: str | None = None) -> str:
    """(Legacy) reset the default session, or reset a given session in place."""
    if session_id and session_id in sessions:
        role = sessions[session_id]["role"]
        create_session(scenario_key, role=role, session_id=session_id)
        sid = session_id
    else:
        create_session(scenario_key, role="duel", session_id=_DEFAULT)
        sid = _DEFAULT
    entry = sessions[sid]
    sim = entry["sim"]
    return (
        f"scenario '{scenario_key}' ready (session_id={sid}, role={entry['role']}): "
        f"{entry['scenario'].name}\n{sim.net.network_summary()}\n"
        f"Objective: {entry['scenario'].objective}"
    )


@mcp.tool()
def list_scenarios() -> str:
    """List available scenario keys."""
    return json.dumps(all_scenario_keys())


@mcp.tool()
def list_sessions() -> str:
    """List live session ids with their scenario + role (no log contents)."""
    return json.dumps(
        {sid: {"scenario": e["scenario"].key, "role": e["role"], "logs": len(e["sim"].logs)}
         for sid, e in sessions.items()}
    )


# --- shared reads (role-gated) ----------------------------------------------

@mcp.tool()
def get_network_state(session_id: str | None = None) -> str:
    """Topology overview + live defense/compromise state (no versions/CVEs)."""
    entry, denied = _check(session_id, "get_network_state")
    if denied:
        return denied
    sim = entry["sim"]
    state = sim.state
    return (
        f"{sim.net.network_summary()}\n\n"
        f"session_id={session_id or _DEFAULT} role={entry['role']} "
        f"compromised={sorted(state['compromised'])} "
        f"isolated={sorted(state['isolated_hosts'])} "
        f"blocked_ips={sorted(state['blocked_ips'])} "
        f"exfiltrated={sorted(state['exfiltrated'])} "
        f"total_logs={len(sim.logs)}"
    )


@mcp.tool()
def get_digest(session_id: str | None = None) -> str:
    """Log statistics + notable events digest (Blue triage input)."""
    entry, denied = _check(session_id, "get_digest")
    if denied:
        return denied
    return entry["sim"].digest()


# --- Red Team ----------------------------------------------------------------

@mcp.tool()
def recon_scan(session_id: str | None = None, target: str = "") -> str:
    """Enumerate a host: services, versions, CVEs, accounts, reachability."""
    entry, denied = _check(session_id, "recon_scan")
    return denied if denied else _red_tool(entry, "recon_scan", {"target": target})


@mcp.tool()
def send_phishing_email(session_id: str | None = None, target_user: str = "", pretext: str = "") -> str:
    """Send a spear-phishing email to a user."""
    entry, denied = _check(session_id, "send_phishing_email")
    return denied if denied else _red_tool(
        entry, "send_phishing_email", {"target_user": target_user, "pretext": pretext})


@mcp.tool()
def brute_force_login(session_id: str | None = None, host: str = "", username: str = "", attempts: int = 5) -> str:
    """Password-guessing against a host login service."""
    entry, denied = _check(session_id, "brute_force_login")
    return denied if denied else _red_tool(
        entry, "brute_force_login", {"host": host, "username": username, "attempts": attempts})


@mcp.tool()
def exploit_known_vulnerability(session_id: str | None = None, host: str = "", cve: str = "") -> str:
    """Attempt to exploit a CVE on a host."""
    entry, denied = _check(session_id, "exploit_known_vulnerability")
    return denied if denied else _red_tool(
        entry, "exploit_known_vulnerability", {"host": host, "cve": cve})


@mcp.tool()
def escalate_privileges(session_id: str | None = None, host: str = "") -> str:
    """Escalate to root/admin on a compromised host."""
    entry, denied = _check(session_id, "escalate_privileges")
    return denied if denied else _red_tool(entry, "escalate_privileges", {"host": host})


@mcp.tool()
def lateral_movement(session_id: str | None = None, from_host: str = "", to_host: str = "", technique: str = "") -> str:
    """Move laterally using harvested creds or admin access."""
    entry, denied = _check(session_id, "lateral_movement")
    return denied if denied else _red_tool(
        entry, "lateral_movement",
        {"from_host": from_host, "to_host": to_host, "technique": technique})


@mcp.tool()
def exfiltrate_data(session_id: str | None = None, host: str = "", dest_ip: str = "") -> str:
    """Exfiltrate data from a compromised host to an external IP."""
    entry, denied = _check(session_id, "exfiltrate_data")
    return denied if denied else _red_tool(entry, "exfiltrate_data", {"host": host, "dest_ip": dest_ip})


@mcp.tool()
def rotate_infrastructure(session_id: str | None = None, reason: str = "") -> str:
    """Switch active external source IP after a perimeter block."""
    entry, denied = _check(session_id, "rotate_infrastructure")
    return denied if denied else _red_tool(entry, "rotate_infrastructure", {"reason": reason})


@mcp.tool()
def mark_task_complete(session_id: str | None = None, summary: str = "") -> str:
    """Mark the operation objective as achieved."""
    entry, denied = _check(session_id, "mark_task_complete")
    return denied if denied else _red_tool(entry, "mark_task_complete", {"summary": summary})


# --- Blue read-only -----------------------------------------------------------

@mcp.tool()
def search_logs(session_id: str | None = None, keyword: str = "") -> str:
    """Full-text search across sanitized log lines."""
    entry, denied = _check(session_id, "search_logs")
    return denied if denied else _analyst_tool(entry, "search_logs", {"keyword": keyword})


@mcp.tool()
def query_logs(session_id: str | None = None, keyword: str = "") -> str:
    """Alias of search_logs."""
    entry, denied = _check(session_id, "query_logs")
    return denied if denied else _analyst_tool(entry, "search_logs", {"keyword": keyword})


@mcp.tool()
def filter_logs(session_id: str | None = None, event_types: list[str] | None = None, hosts: list[str] | None = None, src_ips: list[str] | None = None) -> str:
    """Filter logs by event types, hosts, and/or source IPs."""
    entry, denied = _check(session_id, "filter_logs")
    return denied if denied else _analyst_tool(
        entry, "filter_logs",
        {"event_types": event_types or [], "hosts": hosts or [], "src_ips": src_ips or []})


@mcp.tool()
def get_log_line(session_id: str | None = None, log_id: str = "") -> str:
    """Fetch a single sanitized log line by ID (e.g. L0042)."""
    entry, denied = _check(session_id, "get_log_line")
    return denied if denied else _analyst_tool(entry, "get_log_line", {"log_id": log_id})


@mcp.tool()
def get_log_stats(session_id: str | None = None, limit: int = 20) -> str:
    """Most common event types, hosts, and source IPs."""
    entry, denied = _check(session_id, "get_log_stats")
    return denied if denied else _analyst_tool(entry, "get_log_stats", {"limit": limit})


@mcp.tool()
def inspect_host(session_id: str | None = None, host: str = "") -> str:
    """Recent log lines for one host (convenience over filter_logs)."""
    entry, denied = _check(session_id, "inspect_host")
    return denied if denied else _analyst_tool(
        entry, "filter_logs", {"event_types": [], "hosts": [host], "src_ips": []})


# --- Blue defenses ------------------------------------------------------------

@mcp.tool()
def block_ip(session_id: str | None = None, ip: str = "") -> str:
    """Perimeter-block an IP (inbound source + outbound destination)."""
    entry, denied = _check(session_id, "block_ip")
    return denied if denied else str(defenses.block_ip(entry["sim"], ip))


@mcp.tool()
def isolate_host(session_id: str | None = None, host: str = "") -> str:
    """Isolate a host; revokes any attacker foothold on it."""
    entry, denied = _check(session_id, "isolate_host")
    return denied if denied else str(defenses.isolate_host(entry["sim"], host))


@mcp.tool()
def reset_credentials(session_id: str | None = None, username: str = "") -> str:
    """Reset a user's credentials, invalidating harvested sessions."""
    entry, denied = _check(session_id, "reset_credentials")
    return denied if denied else str(defenses.reset_credentials(entry["sim"], username))


# --- Resources: Tool = act, Resource = inspect --------------------------------

@mcp.resource("cyberarena://{session_id}/network/state")
def res_network_state(session_id: str) -> str:
    """Read-only topology + live compromise/defense state."""
    return get_network_state(session_id=session_id)


@mcp.resource("cyberarena://{session_id}/logs/current")
def res_logs_current(session_id: str) -> str:
    """Sanitized log lines (no ground truth), capped for MCP transport."""
    entry = _resolve(session_id)
    lines = [render_line(e) for e in entry["sim"].logs.snapshot()][-500:]
    return "\n".join(lines)


@mcp.resource("cyberarena://{session_id}/incident/timeline")
def res_incident_timeline(session_id: str) -> str:
    """Chronological attack-relevant timeline (auth/exploit/lateral/exfil/defense)."""
    entry = _resolve(session_id)
    interesting = {
        "login_failure", "login_success", "phishing_email", "phishing_click",
        "port_scan", "exploit_attempt", "webshell", "privilege_escalation",
        "data_transfer_out", "traffic_dropped", "defensive_action",
        "process_exec",
    }
    lines = [render_line(e) for e in entry["sim"].logs.snapshot() if e["event"] in interesting]
    return "\n".join(lines[-300:])


@mcp.resource("cyberarena://{session_id}/scenario/current")
def res_scenario_current(session_id: str) -> str:
    """Scenario key, name, objective, and role for this session."""
    entry = _resolve(session_id)
    sc = entry["scenario"]
    return json.dumps({
        "session_id": session_id, "role": entry["role"],
        "scenario_key": sc.key, "scenario_name": sc.name,
        "objective": sc.objective, "is_attack": sc.is_attack,
    }, indent=2)


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
