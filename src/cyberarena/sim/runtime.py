"""Simulation runtime: clock, state, log store, digest, snapshot text."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .digest import build_digest
from .log_store import LogStore, render_line
from .network import default_model
from .noise import NoiseGenerator
from .scenarios import Scenario


class Simulation:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        # Per-instance network model (randomized for generated scenarios).
        self.net = scenario.network_model or default_model()
        self.logs = LogStore()
        self.state: dict = {
            "compromised": set(),
            "admin": set(),
            "creds": {},  # username -> attacker ip
            "done": False,
            # Active red-team source IP (rotatable; see rotate_infrastructure).
            "attacker_ip": scenario.config.get("attacker_ip", self.net.attacker_ip),
            # Blue Team defensive state (Phase A of the duel).
            "blocked_ips": set(),     # IPs blocked at the perimeter (src & dst)
            "isolated_hosts": set(),  # hosts removed from the network
            # Duel outcome state (Phase B): success requires ACTUAL exfiltration,
            # not just the red team claiming completion via `done`.
            "exfiltrated": set(),     # hosts data was actually exfiltrated from
            "exfil_dropped": 0,       # exfil attempts dropped by perimeter blocks
        }
        self.clock = datetime(2026, 8, 19, 8, 0, 0, tzinfo=timezone.utc)
        # Baseline benign traffic precedes whatever the Red Team does.
        self._add_noise(minutes=60)

    # -- clock ----------------------------------------------------------
    def tick(self, seconds: int) -> None:
        self.clock += timedelta(seconds=seconds)

    @property
    def ts(self) -> str:
        return self.clock.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

    def add(self, **kw) -> dict:
        return self.logs.add(ts=self.ts, **kw)

    # -- views ----------------------------------------------------------
    def snapshot_text(self) -> str:
        return "\n".join(render_line(e) for e in self.logs.snapshot())

    def digest(self) -> str:
        return build_digest(self.logs.snapshot())

    def attack_log_ids(self) -> list[str]:
        return self.logs.attack_ids()

    def attack_ids_by_type(self) -> dict[str, list[str]]:
        return self.logs.attack_ids_by_type()

    def total_logs(self) -> int:
        return len(self.logs)

    # -- internals ------------------------------------------------------
    def _add_noise(self, minutes: int) -> None:
        NoiseGenerator(self, self.scenario.config.get("noise_seed", 1)).generate(minutes)