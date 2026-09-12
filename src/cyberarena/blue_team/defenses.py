"""Blue Team defensive actions that mutate the simulation state.

Phase A of the red/blue duel: these are plain functions (not yet LLM tools).
Each action ACTUALLY removes red-team capability, because the red team tools
read this state before acting:

  - block_ip(ip)          -> perimeter-blocks an IP as both source and
                             destination, stopping external-origin attacks and
                             egress to that IP.
  - isolate_host(host)    -> removes a host from the network; it can no longer
                             be reached or used as a foothold, and the attacker
                             loses any existing foothold there.
  - reset_credentials(user)-> invalidates a harvested credential so it can no
                             longer be used for lateral movement.

Every applied action is written to the log store as a ``source=soc`` /
``event=defensive_action`` entry (no ground-truth tag, so it never counts as
attack evidence). In Phase B these become tools available to an autonomous
defender agent that acts between red-team moves.
"""
from __future__ import annotations


def _log_defense(sim: object, host: str, detail: str, action: str) -> None:
    sim.add(
        source="soc", host=host, user="soc-analyst", src_ip="-", dst_ip="-",
        event="defensive_action", detail=detail, action=action,
    )


def block_ip(sim: object, ip: str) -> str:
    """Block an IP at the perimeter (inbound source and outbound destination)."""
    if ip in sim.state["blocked_ips"]:
        return f"IP {ip} is already blocked at the perimeter."
    sim.tick(30)
    sim.state["blocked_ips"].add(ip)
    gateway = "edge01" if "edge01" in sim.net.hosts else "-"
    _log_defense(
        sim, gateway,
        f"perimeter block applied to IP {ip} (inbound and outbound)", "block_ip",
    )
    return f"DEFENSE: blocked IP {ip} at the perimeter (inbound and outbound)."


def isolate_host(sim: object, host: str) -> str:
    """Isolate a host from the network; the attacker loses any foothold on it."""
    if host not in sim.net.hosts:
        return f"unknown host '{host}'."
    if host in sim.state["isolated_hosts"]:
        return f"{host} is already isolated."
    sim.tick(30)
    sim.state["isolated_hosts"].add(host)
    sim.state["compromised"].discard(host)
    sim.state["admin"].discard(host)
    _log_defense(
        sim, host,
        f"host {host} isolated from the network; any foothold revoked", "isolate_host",
    )
    return f"DEFENSE: isolated {host}; it is unreachable and any foothold there is lost."


def reset_credentials(sim: object, user: str) -> str:
    """Reset a user's credentials, invalidating any harvested session."""
    removed = sim.state["creds"].pop(user, None)
    if removed is not None:
        sim.tick(30)
        dc = "dc01" if "dc01" in sim.net.hosts else "-"
        _log_defense(
            sim, dc,
            f"credentials reset for user '{user}'; harvested session invalidated",
            "reset_credentials",
        )
        return f"DEFENSE: reset credentials for '{user}'; harvested session invalidated."
    return f"No active harvested credentials found for '{user}'."
