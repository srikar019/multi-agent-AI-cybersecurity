"""Synthetic network topology plus a real host/user vulnerability model.

Design goal: attack success must EMERGE from host and account attributes that
the red team discovers via recon -- not from a scenario-level answer key.

The base templates (``HOSTS``, ``ACCOUNTS``, ``REACHABILITY``) describe one
concrete network. A ``NetworkModel`` wraps a (possibly randomized) copy of
those templates so each generated scenario can carry its own vulnerability /
credential / exposure placement. The scenario generator mutates a model; the
fixed reference scenarios use ``default_model()``.

Each host carries:
  - exposed: reachable from the internet (otherwise only via a foothold)
  - services: name -> {version, vulns[]}  (versions/CVEs revealed by recon)
  - local_privesc: whether a local privilege-escalation vector exists

Each account carries:
  - password: weak (guessable) | strong
  - mfa: bool (blocks pure password brute force)
  - awareness: untrained | trained | n/a  (drives phishing success)

``reachability`` encodes network segmentation: which hosts a compromised host
can reach. The "external" key lists the internet-facing hosts.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

# External (internet) addresses. 198.51.100.x / 203.0.113.x are TEST-NET ranges.
ATTACKER_IP = "198.51.100.23"
# The red team's external infrastructure pool. When the defender perimeter-
# blocks one address the attacker can rotate to the next (see the
# rotate_infrastructure tool). Sized so a well-adapted attacker has room to
# maneuver: the defender must spend an action per address it wants to burn.
ATTACKER_POOL = [
    ATTACKER_IP,
    "198.51.100.47",
    "198.51.100.63",
    "198.51.100.81",
    "198.51.100.95",
    "198.51.100.109",
]
BENIGN_EXTERNAL = ["203.0.113.5", "203.0.113.9", "198.51.100.88"]

# Attempts needed to guess a weak password.
WEAK_PASSWORD_THRESHOLD = 5

# --- base templates --------------------------------------------------------
ACCOUNTS: dict[str, dict] = {
    "alice":      {"password": "weak",   "mfa": False, "awareness": "untrained"},
    "bob":        {"password": "strong", "mfa": True,  "awareness": "trained"},
    "carol":      {"password": "strong", "mfa": False, "awareness": "trained"},
    "dave":       {"password": "weak",   "mfa": False, "awareness": "untrained"},
    "svc-backup": {"password": "weak",   "mfa": False, "awareness": "n/a"},
}

HOSTS: dict[str, dict] = {
    "mail01": {
        "ip": "10.0.0.10", "role": "mail_server", "exposed": True,
        "local_privesc": False,
        "services": {
            "smtp":    {"version": "Postfix 3.4",   "vulns": []},
            "imap":    {"version": "Dovecot 2.3",   "vulns": []},
            "webmail": {"version": "Roundcube 1.4", "vulns": ["CVE-2021-44026"]},
        },
        "mailboxes": ["alice", "bob", "carol", "dave"],
    },
    "web01": {
        "ip": "10.0.0.11", "role": "web_server", "exposed": True,
        "local_privesc": True,
        "services": {
            "http":  {"version": "nginx 1.18",    "vulns": []},
            "https": {"version": "nginx 1.18",    "vulns": []},
            "ssh":   {"version": "OpenSSH 8.2",   "vulns": []},
            "app":   {"version": "CustomApp 2.1", "vulns": ["CVE-2023-1234"]},
        },
        "local_users": ["svc-backup"],
    },
    "db01": {
        "ip": "10.0.0.12", "role": "database_server", "exposed": False,
        "local_privesc": False,
        "services": {
            "mssql":  {"version": "MSSQL 2019", "vulns": []},
            "backup": {"version": "Veeam 11",   "vulns": []},
        },
        "local_users": ["svc-backup"],
    },
    "ws-alice": {
        "ip": "10.0.1.20", "role": "workstation", "exposed": False,
        "local_privesc": False, "services": {}, "primary_user": "alice",
    },
    "ws-bob": {
        "ip": "10.0.1.21", "role": "workstation", "exposed": False,
        "local_privesc": False, "services": {}, "primary_user": "bob",
    },
    "dc01": {
        "ip": "10.0.1.5", "role": "domain_controller", "exposed": False,
        "local_privesc": False,
        "services": {
            "ldap":     {"version": "AD LDS",      "vulns": []},
            "kerberos": {"version": "Kerberos v5", "vulns": []},
        },
    },
    "edge01": {
        "ip": "10.0.0.1", "role": "gateway", "exposed": True,
        "local_privesc": False, "services": {},
    },
}

REACHABILITY: dict[str, list[str]] = {
    "external": ["mail01", "web01"],
    "mail01":   ["dc01", "ws-alice", "ws-bob"],
    "web01":    ["db01", "dc01"],
    "ws-alice": ["mail01", "web01", "db01", "dc01"],
    "ws-bob":   ["mail01", "web01", "db01", "dc01"],
    "db01":     ["dc01"],
    "dc01":     ["mail01", "web01", "db01", "ws-alice", "ws-bob"],
}


# --- per-instance model ----------------------------------------------------
@dataclass
class NetworkModel:
    hosts: dict
    accounts: dict
    reachability: dict
    attacker_ip: str = ATTACKER_IP
    attacker_pool: list = field(default_factory=lambda: list(ATTACKER_POOL))
    benign_external: list = field(default_factory=lambda: list(BENIGN_EXTERNAL))
    weak_password_threshold: int = WEAK_PASSWORD_THRESHOLD

    # derived views
    @property
    def users(self) -> list[str]:
        return list(self.accounts)

    @property
    def ip_by_name(self) -> dict[str, str]:
        return {n: h["ip"] for n, h in self.hosts.items()}

    @property
    def name_by_ip(self) -> dict[str, str]:
        return {h["ip"]: n for n, h in self.hosts.items()}

    @property
    def workstations(self) -> list[str]:
        return [n for n, h in self.hosts.items() if h.get("role") == "workstation"]

    @property
    def workstation_user(self) -> dict[str, str]:
        return {
            n: h["primary_user"]
            for n, h in self.hosts.items()
            if h.get("role") == "workstation" and "primary_user" in h
        }

    @property
    def exposed_hosts(self) -> list[str]:
        return [n for n, h in self.hosts.items() if h.get("exposed")]

    @property
    def internal_data_hosts(self) -> list[str]:
        """Internal hosts that hold valuable data (candidate exfil targets)."""
        return [
            n for n, h in self.hosts.items()
            if not h.get("exposed") and h.get("role") in {"database_server", "domain_controller"}
        ]

    def ip(self, name: str) -> str:
        return self.ip_by_name.get(name, "0.0.0.0")

    def host_vulns(self, name: str) -> list[str]:
        out: list[str] = []
        for svc in self.hosts.get(name, {}).get("services", {}).values():
            out.extend(svc.get("vulns", []))
        return out

    def can_reach(self, compromised: set, host: str) -> bool:
        """Is `host` reachable from the attacker's current position?"""
        if host not in self.hosts:
            return False
        if self.hosts[host].get("exposed"):
            return True
        for foothold in compromised:
            if host in self.reachability.get(foothold, []):
                return True
        return False

    def network_summary(self) -> str:
        """Topology overview for agent prompts. Omits versions/CVEs -- those must
        be discovered via recon_scan."""
        lines = [
            "SYNTHETIC NETWORK",
            "internal subnet: 10.0.0.0/24 (servers), 10.0.1.0/24 (clients)",
            "domain: corp.local",
        ]
        for name, info in self.hosts.items():
            svc = ",".join(info.get("services", {}))
            zone = "internet-facing" if info.get("exposed") else "internal"
            lines.append(
                f"  {name:<10} {info['ip']:<16} role={info['role']} [{zone}]"
                + (f" services={svc}" if svc else "")
            )
        lines.append(
            f"users: {', '.join(self.users)} | external attacker IP: {self.attacker_ip}"
        )
        lines.append(
            "Note: service versions, vulnerabilities, and credential properties are "
            "NOT listed here. Use recon_scan to discover them."
        )
        return "\n".join(lines)


def default_model() -> NetworkModel:
    """A fresh copy of the base network (used by the fixed reference scenarios)."""
    return NetworkModel(
        hosts=copy.deepcopy(HOSTS),
        accounts=copy.deepcopy(ACCOUNTS),
        reachability=copy.deepcopy(REACHABILITY),
    )


# --- backward-compatible module-level helpers (operate on the base model) ---
USERS = list(ACCOUNTS)
IP_BY_NAME = {name: info["ip"] for name, info in HOSTS.items()}
NAME_BY_IP = {info["ip"]: name for name, info in HOSTS.items()}
WORKSTATIONS = ["ws-alice", "ws-bob"]
WORKSTATION_USER = {"ws-alice": "alice", "ws-bob": "bob"}


def host_by_name(name: str) -> dict | None:
    return HOSTS.get(name)


def ip_by_name(name: str) -> str:
    return IP_BY_NAME[name]


def name_by_ip(ip: str) -> str | None:
    return NAME_BY_IP.get(ip)


def host_vulns(name: str) -> list[str]:
    info = HOSTS.get(name, {})
    out: list[str] = []
    for svc in info.get("services", {}).values():
        out.extend(svc.get("vulns", []))
    return out


def network_summary() -> str:
    return default_model().network_summary()
