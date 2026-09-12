"""Read-only view over sanitized log entries used by Blue Team search tools."""
from __future__ import annotations

from collections import Counter

from ..sim.log_store import render_line


class LogView:
    def __init__(self, entries: list[dict]) -> None:
        self.entries = entries

    def _lines(self, subset: list[dict] | None = None) -> list[str]:
        return [render_line(e) for e in (subset if subset is not None else self.entries)]

    def search(self, keyword: str, limit: int = 200) -> list[str]:
        kw = keyword.lower()
        hits = [e for e in self.entries if kw in render_line(e).lower()]
        return self._lines(hits)[:limit]

    def filter(
        self,
        event_types: list[str] | None = None,
        hosts: list[str] | None = None,
        src_ips: list[str] | None = None,
        limit: int = 200,
    ) -> list[str]:
        event_types = [s.lower() for s in (event_types or [])]
        hosts = [s.lower() for s in (hosts or [])]
        src_ips = [s.lower() for s in (src_ips or [])]
        hits = [
            e
            for e in self.entries
            if (not event_types or e["event"].lower() in event_types)
            and (not hosts or e["host"].lower() in hosts)
            and (not src_ips or e["src_ip"].lower() in src_ips)
        ]
        return self._lines(hits)[:limit]

    def line(self, log_id: str) -> str:
        lid = log_id.upper()
        for e in self.entries:
            if e["id"] == lid:
                return render_line(e)
        return f"no log line found for {log_id}"

    def stats(self, limit: int = 20) -> str:
        event_counts = Counter((e["source"], e["event"]) for e in self.entries)
        host_counts = Counter(e["host"] for e in self.entries)
        src_counts = Counter(e["src_ip"] for e in self.entries)
        lines = [f"total lines: {len(self.entries)}", "", "events:"]
        for (src, ev), n in event_counts.most_common(limit):
            lines.append(f"  {src}/{ev}: {n}")
        lines.append("")
        lines.append("hosts:")
        for h, n in host_counts.most_common(limit):
            lines.append(f"  {h}: {n}")
        lines.append("")
        lines.append("source ips:")
        for ip, n in src_counts.most_common(limit):
            lines.append(f"  {ip}: {n}")
        return "\n".join(lines)