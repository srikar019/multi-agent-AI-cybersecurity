"""Build a concise digest of the log store for the Triage agent."""
from __future__ import annotations

from collections import Counter

from .log_store import render_line

NOTABLE_EVENTS = {
    "login_failure",
    "phishing_email",
    "phishing_click",
    "exploit_attempt",
    "webshell",
    "privilege_escalation",
    "data_transfer_out",
    "port_scan",
    "account_created",
    "suspicious_transfer",
}


def build_digest(entries: list[dict], sample: int = 120) -> str:
    if not entries:
        return "Log store is empty."

    total = len(entries)
    event_counts = Counter((e["source"], e["event"]) for e in entries)
    host_counts = Counter(e["host"] for e in entries)
    src_counts = Counter(e["src_ip"] for e in entries)
    notable = sorted(
        (e for e in entries if e["event"] in NOTABLE_EVENTS),
        key=lambda e: e["ts"],
    )

    lines = [f"TOTAL LOG LINES: {total}", "", "EVENT COUNTS (source/event: n):"]
    for (src, ev), n in event_counts.most_common(25):
        lines.append(f"  {src}/{ev}: {n}")

    lines.append("")
    lines.append("TOP HOSTS:")
    for host, n in host_counts.most_common(8):
        lines.append(f"  {host}: {n}")

    lines.append("")
    lines.append("TOP SOURCE IPs:")
    for ip, n in src_counts.most_common(8):
        lines.append(f"  {ip}: {n}")

    lines.append("")
    lines.append(f"NOTABLE EVENTS (up to {sample}):")
    if not notable:
        lines.append("  (none)")
    for e in notable[:sample]:
        lines.append("  " + render_line(e))

    return "\n".join(lines)