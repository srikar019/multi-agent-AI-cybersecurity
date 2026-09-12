"""Generate benign background traffic so detection is non-trivial.

The generator is driven by the simulation's ``NetworkModel`` so it works for
both the fixed reference network and randomized generated instances.
"""
from __future__ import annotations

import random


class NoiseGenerator:
    """Emits plausible-but-benign log lines, one virtual minute at a time."""

    def __init__(self, sim: object, seed: int = 1) -> None:
        self.sim = sim
        self.net = sim.net
        self.rng = random.Random(seed)
        # Pick stable reference hosts if they exist in this model.
        self.web_host = "web01" if "web01" in self.net.hosts else (
            self.net.exposed_hosts[0] if self.net.exposed_hosts else None
        )
        self.mail_host = "mail01" if "mail01" in self.net.hosts else None
        self.db_host = "db01" if "db01" in self.net.hosts else None
        self.dc_host = "dc01" if "dc01" in self.net.hosts else None
        self.edge_host = "edge01" if "edge01" in self.net.hosts else None
        self.workstations = self.net.workstations

    def generate(self, minutes: int) -> None:
        for m in range(minutes):
            self.sim.tick(60)
            self._minute(m)

    def _minute(self, m: int) -> None:
        r = self.rng
        add = self.sim.add
        ip = self.net.ip

        # 1-2 web requests, mostly from internal workstations.
        if self.web_host:
            for _ in range(r.randint(1, 2)):
                if self.workstations and r.random() < 0.8:
                    ws = r.choice(self.workstations)
                    src = ip(ws)
                    user = self.net.workstation_user.get(ws, "-")
                else:
                    src = r.choice(self.net.benign_external)
                    user = "-"
                path = r.choice(["home", "docs", "api", "assets", "login", "report"])
                add(
                    source="web", host=self.web_host, user=user, src_ip=src,
                    dst_ip=ip(self.web_host), event="http_request",
                    detail=f"GET /{path} HTTP/1.1 200",
                )

        # Internal email every ~3 min.
        if self.mail_host and m % 3 == 0 and len(self.net.users) >= 2:
            sender, recipient = r.sample(self.net.users, 2)
            subject = r.choice(
                ["Quarterly report", "Lunch menu", "Project update", "Meeting notes"]
            )
            add(
                source="email", host=self.mail_host, user=recipient,
                src_ip=ip(self.mail_host), dst_ip=ip(self.mail_host),
                event="email_internal",
                detail=f"internal mail {sender}->{recipient} subject='{subject}'",
            )

        # Workstation login every ~5 min.
        if self.workstations and m % 5 == 0:
            ws = r.choice(self.workstations)
            add(
                source="auth", host=ws, user=self.net.workstation_user.get(ws, "-"),
                src_ip=ip(ws), dst_ip=ip(ws), event="login_success",
                detail="interactive logon with smartcard",
            )

        # Benign failed login (user error) every ~7 min.
        if self.workstations and m % 7 == 0:
            ws = r.choice(self.workstations)
            add(
                source="auth", host=ws, user=self.net.workstation_user.get(ws, "-"),
                src_ip=ip(ws), dst_ip=ip(ws), event="login_failure",
                detail="incorrect password (benign user error)",
            )

        # Outbound HTTPS to a known-good SaaS provider every ~6 min.
        if self.workstations and m % 6 == 0:
            ws = r.choice(self.workstations)
            add(
                source="firewall", host=ws, user=self.net.workstation_user.get(ws, "-"),
                src_ip=ip(ws), dst_ip=r.choice(self.net.benign_external),
                event="https_out", detail="TLS to known-good SaaS provider",
            )

        # DNS + firewall allow every ~10 min.
        if m % 10 == 0:
            if self.workstations and self.dc_host:
                ws = r.choice(self.workstations)
                add(
                    source="dns", host=ws, user="-", src_ip=ip(ws), dst_ip=ip(self.dc_host),
                    event="dns_query",
                    detail=f"resolve {r.choice(['corp.local', 'office365.com', 'updates.microsoft.com'])}",
                )
            if self.edge_host and self.web_host:
                add(
                    source="firewall", host=self.edge_host, user="-", src_ip="0.0.0.0",
                    dst_ip=ip(self.web_host), event="allow",
                    detail="established session (benign)",
                )

        # Backup job every ~15 min.
        if self.db_host and m % 15 == 0:
            add(
                source="process", host=self.db_host, user="svc-backup",
                src_ip=ip(self.db_host), dst_ip=ip(self.db_host),
                event="backup_job", detail="nightly backup to vault completed",
            )
