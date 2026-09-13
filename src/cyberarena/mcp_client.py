"""In-process MCP client: LangChain tools routed THROUGH the MCP server.

This is what makes MCP the real tool layer instead of an unused adapter:

    Red Agent  -> make_mcp_red_tools(session_id)  -> mcp_server.* -> sim
    Blue Agent -> make_mcp_blue_tools(session_id) -> mcp_server.* -> sim

Same process (no stdio/HTTP overhead, eval stays fast), but every call
passes through MCP role checks + session isolation + resources.

For a true out-of-process client (Claude Desktop, separate agent service),
run ``python -m cyberarena.mcp_server --transport streamable-http`` and use
``langchain_mcp_adapters`` against the same tool names.
"""
from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from . import mcp_server
from . import config


# --- schemas mirror red_team/tools.py + blue_team/agents.py ------------------

class _Target(BaseModel):
    target: str = Field(default="", description="Hostname to scan")


class _Phish(BaseModel):
    target_user: str = Field(description="Username to phish")
    pretext: str = Field(description="One-line pretext")


class _Brute(BaseModel):
    host: str = ""
    username: str = ""
    attempts: int = 5


class _Exploit(BaseModel):
    host: str = ""
    cve: str = ""


class _Host(BaseModel):
    host: str = ""


class _Lateral(BaseModel):
    from_host: str = ""
    to_host: str = ""
    technique: str = ""


class _Exfil(BaseModel):
    host: str = ""
    dest_ip: str = ""


class _Summary(BaseModel):
    summary: str = ""


class _Reason(BaseModel):
    reason: str = ""


class _Keyword(BaseModel):
    keyword: str = ""


class _Filter(BaseModel):
    event_types: list[str] = Field(default_factory=list)
    hosts: list[str] = Field(default_factory=list)
    src_ips: list[str] = Field(default_factory=list)


class _Line(BaseModel):
    log_id: str = ""


class _Stats(BaseModel):
    limit: int = 20


class _Ip(BaseModel):
    ip: str = ""


class _Username(BaseModel):
    username: str = ""


def make_mcp_red_tools(session_id: str) -> list:
    """Red attack tools bound to one MCP session (role must allow them)."""

    @tool("recon_scan", args_schema=_Target)
    def recon_scan(target: str = "") -> str:
        """Enumerate a host: services, versions, CVEs, accounts, reachability."""
        return mcp_server.recon_scan(session_id=session_id, target=target)

    @tool("send_phishing_email", args_schema=_Phish)
    def send_phishing_email(target_user: str, pretext: str) -> str:
        """Send a spear-phishing email to a user."""
        return mcp_server.send_phishing_email(
            session_id=session_id, target_user=target_user, pretext=pretext)

    @tool("brute_force_login", args_schema=_Brute)
    def brute_force_login(host: str, username: str, attempts: int = 5) -> str:
        """Password-guessing against a host login service."""
        return mcp_server.brute_force_login(
            session_id=session_id, host=host, username=username, attempts=attempts)

    @tool("exploit_known_vulnerability", args_schema=_Exploit)
    def exploit_known_vulnerability(host: str, cve: str) -> str:
        """Attempt to exploit a CVE on a host."""
        return mcp_server.exploit_known_vulnerability(
            session_id=session_id, host=host, cve=cve)

    @tool("escalate_privileges", args_schema=_Host)
    def escalate_privileges(host: str) -> str:
        """Escalate to root/admin on a compromised host."""
        return mcp_server.escalate_privileges(session_id=session_id, host=host)

    @tool("lateral_movement", args_schema=_Lateral)
    def lateral_movement(from_host: str, to_host: str, technique: str) -> str:
        """Move laterally using harvested creds or admin access."""
        return mcp_server.lateral_movement(
            session_id=session_id, from_host=from_host, to_host=to_host, technique=technique)

    @tool("exfiltrate_data", args_schema=_Exfil)
    def exfiltrate_data(host: str, dest_ip: str) -> str:
        """Exfiltrate data from a compromised host to an external IP."""
        return mcp_server.exfiltrate_data(session_id=session_id, host=host, dest_ip=dest_ip)

    @tool("rotate_infrastructure", args_schema=_Reason)
    def rotate_infrastructure(reason: str) -> str:
        """Switch active external source IP after a perimeter block."""
        return mcp_server.rotate_infrastructure(session_id=session_id, reason=reason)

    @tool("mark_task_complete", args_schema=_Summary)
    def mark_task_complete(summary: str) -> str:
        """Mark the operation objective as achieved."""
        return mcp_server.mark_task_complete(session_id=session_id, summary=summary)

    return [
        recon_scan, send_phishing_email, brute_force_login,
        exploit_known_vulnerability, escalate_privileges, lateral_movement,
        exfiltrate_data, rotate_infrastructure, mark_task_complete,
    ]


def make_mcp_blue_tools(session_id: str) -> list:
    """Blue read-only log tools bound to one MCP session."""

    @tool("search_logs", args_schema=_Keyword)
    def search_logs(keyword: str) -> str:
        """Full-text search across sanitized log lines."""
        return mcp_server.search_logs(session_id=session_id, keyword=keyword)

    @tool("filter_logs", args_schema=_Filter)
    def filter_logs(event_types: list[str], hosts: list[str], src_ips: list[str]) -> str:
        """Filter logs by event types, hosts, and/or source IPs."""
        return mcp_server.filter_logs(
            session_id=session_id, event_types=event_types, hosts=hosts, src_ips=src_ips)

    @tool("get_log_stats", args_schema=_Stats)
    def get_log_stats(limit: int = 20) -> str:
        """Most common event types, hosts, and source IPs."""
        return mcp_server.get_log_stats(session_id=session_id, limit=limit)

    @tool("get_log_line", args_schema=_Line)
    def get_log_line(log_id: str) -> str:
        """Fetch a single sanitized log line by ID."""
        return mcp_server.get_log_line(session_id=session_id, log_id=log_id)

    return [search_logs, filter_logs, get_log_stats, get_log_line]


def make_mcp_defender_tools(session_id: str) -> list:
    """Blue defender tools (reads + real defenses) bound to one MCP session."""

    @tool("block_ip", args_schema=_Ip)
    def block_ip_tool(ip: str) -> str:
        """Perimeter-block an IP (inbound + outbound)."""
        return mcp_server.block_ip(session_id=session_id, ip=ip)

    @tool("isolate_host", args_schema=_Host)
    def isolate_host_tool(host: str) -> str:
        """Isolate a host; revokes any attacker foothold on it."""
        return mcp_server.isolate_host(session_id=session_id, host=host)

    @tool("reset_credentials", args_schema=_Username)
    def reset_credentials_tool(username: str) -> str:
        """Reset a user's credentials, invalidating harvested sessions."""
        return mcp_server.reset_credentials(session_id=session_id, username=username)

    return [block_ip_tool, isolate_host_tool, reset_credentials_tool, *make_mcp_blue_tools(session_id)]


# --- agent builders THROUGH MCP (agents operate via MCP, not direct funcs) ---

def build_red_agent_via_mcp(llm, session_id: str, max_steps: int | None = None):
    """ReAct red agent whose tools are MCP calls for this session."""
    from langgraph.prebuilt import create_react_agent

    from .red_team.tools import red_environment_context

    max_steps = max_steps or config.MAX_RED_STEPS
    entry = mcp_server.sessions[session_id]
    prompt = (
        "You are an autonomous RED TEAM agent operating inside a fully SIMULATED "
        "corporate network via MCP tools. Always act through your tools.\n\n"
        f"{red_environment_context(entry['scenario'], entry['sim'].net)}\n"
        f"You have at most {max_steps} tool calls."
    )
    return create_react_agent(llm, make_mcp_red_tools(session_id), prompt=prompt)


def build_analyst_agent_via_mcp(llm, session_id: str):
    """Blue analyst ReAct agent whose log tools are MCP calls."""
    from langgraph.prebuilt import create_react_agent

    from .blue_team.agents import ANALYST_SYSTEM

    return create_react_agent(llm, make_mcp_blue_tools(session_id), prompt=ANALYST_SYSTEM)


def build_defender_agent_via_mcp(llm, session_id: str, max_actions: int = 2):
    """Blue defender agent (reads + defenses) operating via MCP."""
    from langgraph.prebuilt import create_react_agent

    from .blue_team.defender import DEFENDER_SYSTEM

    entry = mcp_server.sessions[session_id]
    return create_react_agent(
        llm, make_mcp_defender_tools(session_id),
        prompt=DEFENDER_SYSTEM.format(
            context=entry["sim"].net.network_summary(), max_actions=max_actions),
    )
