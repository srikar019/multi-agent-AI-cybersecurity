"""Blue Team defender agent for the interleaved red/blue duel (Phase B).

The defender acts between red-team moves. It inspects the sanitized log view
with read-only search tools and may apply real defensive actions
(block_ip / isolate_host / reset_credentials) that immediately mutate the
simulation state. Applied actions are measured from the log store, not from
the model's claims.
"""
from __future__ import annotations

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from .agents import make_analyst_tools
from .defenses import block_ip, isolate_host, reset_credentials
from .log_view import LogView

DEFENDER_SYSTEM = """You are the BLUE TEAM defender on duty in a simulated SOC. An adversary may be
actively attacking this network, and you act between attacker moves -- any
defensive action you take takes effect immediately in the simulation.

Environment:
{context}

Your defensive tools ACTUALLY remove attacker capability:
- block_ip: perimeter-blocks an IP (both inbound source and outbound
  destination). Use against external IPs behind scans, phishing, brute force,
  or exploit attempts, and against suspicious egress destinations.
- isolate_host: removes a host from the network and revokes any attacker
  foothold on it. Use on hosts with clear signs of compromise.
- reset_credentials: invalidates a user's credentials and any harvested
  session. Use after phishing clicks, credential theft, or brute-force
  successes for that user.

You also have read-only log tools (search_logs, filter_logs, get_log_stats,
get_log_line). Ground every action in log evidence and cite the justifying
log IDs.

Rules:
- Prefer the least disruptive effective action.
- Do not react to ordinary benign noise; require real evidence of an attack.
- Take at most {max_actions} defensive actions this turn.
- If no action is warranted, reply with exactly: PASS"""


class BlockIpSchema(BaseModel):
    ip: str = Field(description="IP address to block at the perimeter")


class IsolateHostSchema(BaseModel):
    host: str = Field(description="Hostname to isolate from the network")


class ResetCredsSchema(BaseModel):
    username: str = Field(description="Username whose credentials to reset")


def make_defender_tools(sim: object, view: LogView) -> list:
    @tool("block_ip", args_schema=BlockIpSchema)
    def block_ip_tool(ip: str) -> str:
        """Perimeter-block an IP (inbound source and outbound destination)."""
        return block_ip(sim, ip)

    @tool("isolate_host", args_schema=IsolateHostSchema)
    def isolate_host_tool(host: str) -> str:
        """Isolate a host from the network; any attacker foothold on it is revoked."""
        return isolate_host(sim, host)

    @tool("reset_credentials", args_schema=ResetCredsSchema)
    def reset_credentials_tool(username: str) -> str:
        """Reset a user's credentials, invalidating any harvested session."""
        return reset_credentials(sim, username)

    return [
        block_ip_tool,
        isolate_host_tool,
        reset_credentials_tool,
        *make_analyst_tools(view),
    ]


def run_defender_turn(
    llm,
    sim: object,
    *,
    context: str | None = None,
    new_lines: list[str] | None = None,
    max_actions: int = 2,
    recursion_limit: int = 12,
) -> dict:
    """Run one defender turn. Returns {"actions": [...], "error": str|None};
    actions are the defensive_action log entries actually applied."""
    view = LogView(sim.logs.snapshot())
    before_ids = {e["id"] for e in view.entries}
    agent = create_react_agent(
        llm,
        make_defender_tools(sim, view),
        prompt=DEFENDER_SYSTEM.format(
            context=context or sim.net.network_summary(),
            max_actions=max_actions,
        ),
    )
    content = (
        f"The log store currently has {len(view.entries)} lines.\n\n"
        f"Log digest:\n{sim.digest()}\n"
    )
    if new_lines:
        content += "\nNew log lines since your last turn:\n" + "\n".join(new_lines)
    content += "\nReview the activity and take defensive action now if warranted."

    error = None
    try:
        agent.invoke(
            {"messages": [{"role": "user", "content": content}]},
            config={"recursion_limit": recursion_limit},
        )
    except Exception as exc:  # defender failure must not crash the duel
        error = str(exc)

    actions = [
        {
            "ts": e["ts"],
            "detail": e["detail"],
            "action": e.get("action", "unknown"),
        }
        for e in sim.logs.snapshot()
        if e["event"] == "defensive_action" and e["id"] not in before_ids
    ]
    return {"actions": actions, "error": error}
