"""Append-only in-memory log store.

Every entry has a unique ID. Attack-originated entries carry a hidden
`_meta` key with the ground-truth kill-chain phase; this is stripped from
the snapshot the Blue Team sees, and is used only by the evaluator.
"""
from __future__ import annotations

from typing import Any


class LogStore:
    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []
        self._n = 0

    def add(
        self,
        *,
        ts: str,
        source: str,
        host: str,
        user: str,
        src_ip: str,
        dst_ip: str,
        event: str,
        detail: str,
        gt: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        self._n += 1
        entry: dict[str, Any] = {
            "id": f"L{self._n:04d}",
            "ts": ts,
            "source": source,
            "host": host,
            "user": user,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "event": event,
            "detail": detail,
            **extra,
        }
        if gt:
            entry["_meta"] = {"gt": gt}
        self._entries.append(entry)
        return entry

    def snapshot(self) -> list[dict[str, Any]]:
        """View of logs without hidden ground-truth metadata."""
        return [
            {k: v for k, v in e.items() if not k.startswith("_")}
            for e in self._entries
        ]

    def attack_ids(self) -> list[str]:
        return [e["id"] for e in self._entries if e.get("_meta")]

    def attack_ids_by_type(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for e in self._entries:
            gt = e.get("_meta", {}).get("gt")
            if gt:
                out.setdefault(gt, []).append(e["id"])
        return out

    def __len__(self) -> int:
        return len(self._entries)


def render_line(e: dict[str, Any]) -> str:
    return (
        f"{e['id']} | {e['ts']} | {e['source']:>8} | host={e['host']:<12} "
        f"user={e['user']:<12} src={e['src_ip']:<15} dst={e['dst_ip']:<15} "
        f"event={e['event']} | {e['detail']}"
    )