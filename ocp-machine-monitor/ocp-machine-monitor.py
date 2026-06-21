#!/usr/bin/env python3
"""
OCP Machine Monitor — watches OpenShift Machines by instance type (or
MachineSet name) and optionally sends Slack notifications on phase transitions.

Usage:
    python3 ocp-machine-monitor.py -i p5.48xlarge
    python3 ocp-machine-monitor.py -i p5.48xlarge -i g5.12xlarge
    python3 ocp-machine-monitor.py -m my-machineset-name
    python3 ocp-machine-monitor.py -i p5.48xlarge -s https://hooks.slack.com/services/XXX
    python3 ocp-machine-monitor.py -i p5.48xlarge -v -p 15 -l monitor.log
    python3 ocp-machine-monitor.py -i p5.48xlarge --json --once
    python3 ocp-machine-monitor.py -i p5.48xlarge --timeout 60 --min-running 3
    SLACK_WEBHOOK_URL=https://... python3 ocp-machine-monitor.py -i p5.48xlarge
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

# ── Machine phases (from openshift/api machine/v1beta1) ──────────────────────

class Phase(str, Enum):
    PROVISIONING = "Provisioning"
    PROVISIONED = "Provisioned"
    RUNNING = "Running"
    FAILED = "Failed"
    DELETING = "Deleting"
    UNKNOWN = "Unknown"
    NOT_FOUND = "NotFound"

    @property
    def is_settled(self) -> bool:
        """Terminal or actively being removed — no further forward progress expected."""
        return self in (Phase.RUNNING, Phase.FAILED, Phase.DELETING)

    @property
    def emoji(self) -> str:
        return {
            Phase.PROVISIONING: ":hourglass_flowing_sand:",
            Phase.PROVISIONED: ":large_blue_circle:",
            Phase.RUNNING: ":white_check_mark:",
            Phase.FAILED: ":x:",
            Phase.DELETING: ":wastebasket:",
            Phase.UNKNOWN: ":question:",
            Phase.NOT_FOUND: ":grey_question:",
        }[self]

    @classmethod
    def from_str(cls, value: str | None) -> "Phase":
        if not value:
            return cls.UNKNOWN
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN


# ── ANSI colors ──────────────────────────────────────────────────────────────

class Color:
    _enabled = True

    _RESET = "\033[0m"
    _BOLD = "\033[1m"
    _DIM = "\033[2m"
    _RED = "\033[31m"
    _GREEN = "\033[32m"
    _YELLOW = "\033[33m"
    _BLUE = "\033[34m"
    _MAGENTA = "\033[35m"
    _CYAN = "\033[36m"
    _WHITE = "\033[37m"
    _BG_GREEN = "\033[42m"
    _BG_RED = "\033[41m"
    _BG_YELLOW = "\033[43m"

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def _c(cls, code: str) -> str:
        return code if cls._enabled else ""

    RESET = property(lambda self: Color._c(Color._RESET))
    BOLD = property(lambda self: Color._c(Color._BOLD))
    DIM = property(lambda self: Color._c(Color._DIM))
    RED = property(lambda self: Color._c(Color._RED))
    GREEN = property(lambda self: Color._c(Color._GREEN))
    YELLOW = property(lambda self: Color._c(Color._YELLOW))
    BLUE = property(lambda self: Color._c(Color._BLUE))
    MAGENTA = property(lambda self: Color._c(Color._MAGENTA))
    CYAN = property(lambda self: Color._c(Color._CYAN))
    WHITE = property(lambda self: Color._c(Color._WHITE))
    BG_GREEN = property(lambda self: Color._c(Color._BG_GREEN))
    BG_RED = property(lambda self: Color._c(Color._BG_RED))
    BG_YELLOW = property(lambda self: Color._c(Color._BG_YELLOW))


C = Color()

PHASE_ICONS = {
    Phase.PROVISIONING: "⏳",
    Phase.PROVISIONED: "🔵",
    Phase.RUNNING: "✅",
    Phase.FAILED: "❌",
    Phase.DELETING: "🗑️ ",
    Phase.UNKNOWN: "❓",
    Phase.NOT_FOUND: "❓",
}

PHASE_COLORS = {
    Phase.PROVISIONING: lambda: Color._c(Color._YELLOW),
    Phase.PROVISIONED: lambda: Color._c(Color._BLUE),
    Phase.RUNNING: lambda: Color._c(Color._GREEN),
    Phase.FAILED: lambda: Color._c(Color._RED),
    Phase.DELETING: lambda: Color._c(Color._MAGENTA),
    Phase.UNKNOWN: lambda: Color._c(Color._DIM),
    Phase.NOT_FOUND: lambda: Color._c(Color._DIM),
}


def colored_phase_str(phase: Phase) -> str:
    color = PHASE_COLORS.get(phase, lambda: "")()
    icon = PHASE_ICONS.get(phase, "")
    return f"{color}{icon}{Color._c(Color._RESET)}"


# ── Slack text escaping ──────────────────────────────────────────────────────

def slack_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class LastOperation:
    description: str = ""
    state: str = ""
    op_type: str = ""
    last_updated: str = ""

    def summary(self) -> str:
        parts = []
        if self.op_type:
            parts.append(self.op_type)
        if self.state:
            parts.append(self.state)
        if self.description:
            parts.append(f"({self.description})")
        return " ".join(parts) if parts else "n/a"


@dataclass
class MachineCondition:
    cond_type: str = ""
    status: str = ""
    reason: str = ""
    message: str = ""


@dataclass
class MachineEvent:
    event_type: str = ""
    reason: str = ""
    message: str = ""
    count: int = 0
    last_seen: str = ""


@dataclass
class MachineInfo:
    name: str = ""
    phase: Phase = Phase.UNKNOWN
    error_message: str = ""
    error_reason: str = ""
    node_name: str = ""
    node_ready: bool = False
    provider_id: str = ""
    instance_id: str = ""
    addresses: list[str] = field(default_factory=list)
    last_operation: LastOperation | None = None
    conditions: list[MachineCondition] = field(default_factory=list)
    events: list[MachineEvent] = field(default_factory=list)
    creation_ts: str = ""
    age_str: str = ""
    provision_duration: str = ""


@dataclass
class MachineSetInfo:
    name: str = ""
    desired: int = 0
    ready: int = 0
    available: int = 0
    replicas: int = 0
    error_message: str = ""


@dataclass
class DiscoveryResult:
    machinesets: list[MachineSetInfo] = field(default_factory=list)
    machines: list[str] = field(default_factory=list)
    raw_machine_items: dict = field(default_factory=dict)


# ── Config ───────────────────────────────────────────────────────────────────

@dataclass
class MonitorConfig:
    instance_types: list[str] = field(default_factory=list)
    machineset_names: list[str] = field(default_factory=list)
    webhook_url: str | None = None
    poll_interval: int = 30
    stuck_threshold_minutes: int = 15
    verbose: bool = False
    json_output: bool = False
    once: bool = False
    timeout_minutes: int | None = None
    namespace: str = "openshift-machine-api"
    min_running: int | None = None
    slack_mention: str = "<!channel>"
    dry_run: bool = False
    heartbeat_interval_minutes: int | None = None
    lock_machines: bool = False
    status_file: str | None = None
    log_file: str | None = None
    no_color: bool = False
    oc_max_retries: int = 3  # must be >= 1
    oc_retry_base_delay: float = 2.0

    def __post_init__(self):
        if self.oc_max_retries < 1:
            self.oc_max_retries = 1


# ── Logging setup ────────────────────────────────────────────────────────────

VERBOSE_FMT = "%(asctime)s [%(levelname)-7s] %(message)s"
NORMAL_FMT = "[%(asctime)s] %(message)s"

log = logging.getLogger("gpu-monitor")


def setup_logging(cfg: MonitorConfig) -> None:
    level = logging.DEBUG if cfg.verbose else logging.INFO
    fmt = VERBOSE_FMT if cfg.verbose else NORMAL_FMT

    stream = sys.stderr if cfg.json_output else sys.stdout
    stdout_handler = logging.StreamHandler(stream)
    stdout_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt="%H:%M:%S"))
    log.addHandler(stdout_handler)

    if cfg.log_file:
        parent = Path(cfg.log_file).parent
        if not parent.exists():
            log.warning("Log file parent directory does not exist: %s — creating it", parent)
            parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(cfg.log_file, mode="a")
        file_handler.setFormatter(logging.Formatter(fmt=VERBOSE_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
        log.addHandler(file_handler)

    log.setLevel(level)

    if cfg.no_color or cfg.json_output or not sys.stdout.isatty():
        Color.disable()


# ── Time helpers ─────────────────────────────────────────────────────────────

def parse_k8s_timestamp(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def human_duration(seconds: float) -> str:
    """Format seconds as human-readable duration. Negative values indicate clock skew."""
    if seconds < 0:
        return "n/a"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def compute_age(creation_ts: str) -> str:
    dt = parse_k8s_timestamp(creation_ts)
    if dt is None:
        return "n/a"
    delta = datetime.now(timezone.utc) - dt
    return human_duration(delta.total_seconds())


def compute_provision_duration(creation_ts: str) -> str:
    """Duration from machine creation to now (for freshly Running machines)."""
    dt = parse_k8s_timestamp(creation_ts)
    if dt is None:
        return ""
    delta = datetime.now(timezone.utc) - dt
    return human_duration(delta.total_seconds())


# ── OC helpers ───────────────────────────────────────────────────────────────

def check_oc_login() -> bool:
    try:
        result = subprocess.run(
            ["oc", "whoami"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            log.error("%sNot logged in to OpenShift.%s Run 'oc login' first.", C.RED, C.RESET)
            log.debug("oc whoami stderr: %s", result.stderr.strip())
            return False
        user = result.stdout.strip()
        log.info("%sLogged in as:%s %s", C.GREEN, C.RESET, user)

        result = subprocess.run(
            ["oc", "whoami", "--show-server"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            log.info("%sCluster:%s %s", C.CYAN, C.RESET, result.stdout.strip())
        else:
            log.debug("Could not retrieve cluster URL: %s", result.stderr.strip())

        return True
    except FileNotFoundError:
        log.error("'oc' command not found. Install the OpenShift CLI first.")
        return False
    except subprocess.TimeoutExpired:
        log.error("'oc whoami' timed out. Cluster may be unreachable.")
        return False


def _retry_delay(base: float, attempt: int) -> float:
    return base * (2 ** (attempt - 1)) + random.uniform(0, 1)


def oc_get_json(resource: str, cfg: MonitorConfig) -> dict | None:
    cmd = ["oc", "get", resource, "-n", cfg.namespace, "-o", "json"]
    log.debug("Running: %s", " ".join(cmd))
    for attempt in range(1, cfg.oc_max_retries + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                log.warning(
                    "oc command failed (rc=%d, attempt %d/%d): %s",
                    result.returncode, attempt, cfg.oc_max_retries, result.stderr.strip(),
                )
                if attempt < cfg.oc_max_retries:
                    time.sleep(_retry_delay(cfg.oc_retry_base_delay, attempt))
                    continue
                return None
            data = json.loads(result.stdout)
            log.debug("Got %d items from %s", len(data.get("items", [])), resource)
            return data
        except subprocess.TimeoutExpired:
            log.warning("oc command timed out (attempt %d/%d)", attempt, cfg.oc_max_retries)
            if attempt < cfg.oc_max_retries:
                time.sleep(_retry_delay(cfg.oc_retry_base_delay, attempt))
                continue
            return None
        except json.JSONDecodeError as exc:
            log.error("Failed to parse oc output: %s", exc)
            return None
    return None


def oc_get_events(machine_name: str, cfg: MonitorConfig) -> list[MachineEvent]:
    cmd = [
        "oc", "get", "events", "-n", cfg.namespace,
        "--field-selector", f"involvedObject.name={machine_name}",
        "--sort-by=.lastTimestamp", "-o", "json",
    ]
    log.debug("Fetching events for %s", machine_name)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            log.debug("Failed to fetch events for %s: %s", machine_name, result.stderr.strip())
            return []
        data = json.loads(result.stdout)
        events = []
        for item in data.get("items", [])[-5:]:
            events.append(MachineEvent(
                event_type=item.get("type", ""),
                reason=item.get("reason", ""),
                message=item.get("message", ""),
                count=item.get("count", 0),
                last_seen=item.get("lastTimestamp", ""),
            ))
        return events
    except subprocess.TimeoutExpired:
        log.debug("Events fetch timed out for %s", machine_name)
        return []
    except json.JSONDecodeError as exc:
        log.debug("Failed to parse events for %s: %s", machine_name, exc)
        return []


def check_node_ready(node_name: str) -> bool:
    if not node_name:
        return False
    cmd = [
        "oc", "get", "node", node_name,
        "-o", "jsonpath={.status.conditions[?(@.type=='Ready')].status}",
    ]
    log.debug("Checking node readiness: %s", node_name)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0 and result.stdout.strip() == "True"
    except subprocess.TimeoutExpired:
        log.debug("Node readiness check timed out for %s", node_name)
        return False
    except OSError as exc:
        log.debug("Node readiness check failed for %s: %s", node_name, exc)
        return False


# ── Discovery ────────────────────────────────────────────────────────────────

def _parse_machineset_item(item: dict) -> MachineSetInfo:
    status = item.get("status", {})
    spec = item.get("spec", {})
    return MachineSetInfo(
        name=item["metadata"]["name"],
        desired=spec.get("replicas", 0),
        replicas=status.get("replicas", 0),
        ready=status.get("readyReplicas", 0),
        available=status.get("availableReplicas", 0),
        error_message=status.get("errorMessage", ""),
    )


def discover(cfg: MonitorConfig) -> DiscoveryResult | None:
    ms_data = oc_get_json("machinesets", cfg)
    if ms_data is None:
        return None

    machineset_infos: list[MachineSetInfo] = []
    ms_names: list[str] = []

    for item in ms_data.get("items", []):
        name = item["metadata"]["name"]
        it = (
            item.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("providerSpec", {})
            .get("value", {})
            .get("instanceType")
        )
        match = False
        if cfg.instance_types and it in cfg.instance_types:
            match = True
        if cfg.machineset_names and name in cfg.machineset_names:
            match = True
        if not match:
            continue

        ms_names.append(name)
        machineset_infos.append(_parse_machineset_item(item))

    if not ms_names:
        filters = cfg.instance_types + cfg.machineset_names
        log.error("No MachineSets found matching: %s", ", ".join(filters))
        return None

    log.debug("Found %d MachineSet(s): %s", len(ms_names), ", ".join(ms_names))

    m_data = oc_get_json("machines", cfg)
    if m_data is None:
        return None

    machines = []
    raw_items = {}
    for item in m_data.get("items", []):
        owners = item.get("metadata", {}).get("ownerReferences", [])
        for owner in owners:
            if owner.get("kind") == "MachineSet" and owner.get("name") in ms_names:
                mname = item["metadata"]["name"]
                machines.append(mname)
                raw_items[mname] = item
                break

    if not machines:
        log.error("No Machines found belonging to MachineSets: %s", ", ".join(ms_names))
        return None

    log.debug("Found %d machine(s)", len(machines))
    return DiscoveryResult(machinesets=machineset_infos, machines=machines, raw_machine_items=raw_items)


def _extract_instance_id(provider_id: str) -> str:
    if not provider_id:
        return ""
    parts = provider_id.rstrip("/").split("/")
    return parts[-1] if parts else ""


def parse_machine_items(
    machine_names: list[str],
    raw_items: dict,
    cfg: MonitorConfig,
) -> dict[str, MachineInfo]:
    results: dict[str, MachineInfo] = {}

    for name in machine_names:
        item = raw_items.get(name)
        if item is None:
            results[name] = MachineInfo(name=name, phase=Phase.NOT_FOUND)
            continue

        status = item.get("status", {})
        metadata = item.get("metadata", {})
        spec = item.get("spec", {})

        phase = Phase.from_str(status.get("phase"))
        provider_id = spec.get("providerID", "") or status.get("providerStatus", {}).get("instanceId", "")

        lo_raw = status.get("lastOperation")
        last_op = None
        if lo_raw:
            last_op = LastOperation(
                description=lo_raw.get("description", ""),
                state=lo_raw.get("state", ""),
                op_type=lo_raw.get("type", ""),
                last_updated=lo_raw.get("lastUpdated", ""),
            )

        conditions = [
            MachineCondition(
                cond_type=c.get("type", ""), status=c.get("status", ""),
                reason=c.get("reason", ""), message=c.get("message", ""),
            )
            for c in status.get("conditions", [])
        ]

        node_ref = status.get("nodeRef", {})
        node_name = node_ref.get("name", "") if node_ref else ""

        node_ready = False
        if phase == Phase.RUNNING and node_name:
            node_ready = check_node_ready(node_name)

        addresses = [
            f"{a.get('type', '')}: {a.get('address', '')}"
            for a in status.get("addresses", [])
        ]

        creation_ts = metadata.get("creationTimestamp", "")

        provision_duration = ""
        if phase == Phase.RUNNING:
            provision_duration = compute_provision_duration(creation_ts)

        events = []
        if cfg.verbose:
            events = oc_get_events(name, cfg)

        info = MachineInfo(
            name=name, phase=phase,
            error_message=status.get("errorMessage", ""),
            error_reason=status.get("errorReason", ""),
            node_name=node_name, node_ready=node_ready,
            provider_id=provider_id, instance_id=_extract_instance_id(provider_id),
            addresses=addresses, last_operation=last_op,
            conditions=conditions, events=events,
            creation_ts=creation_ts, age_str=compute_age(creation_ts),
            provision_duration=provision_duration,
        )
        results[name] = info
        log.debug(
            "  %s -> phase=%s, node=%s(%s), age=%s",
            name, phase.value, node_name or "n/a",
            "ready" if node_ready else "not-ready", info.age_str,
        )

    return results


# ── Slack ────────────────────────────────────────────────────────────────────

def send_slack(cfg: MonitorConfig, message: str) -> bool:
    if not cfg.webhook_url or cfg.dry_run:
        if cfg.dry_run and cfg.webhook_url:
            log.info("%s[DRY-RUN] Would send Slack message (%d chars)%s", C.DIM, len(message), C.RESET)
        else:
            log.debug("Slack disabled — skipping notification")
        return False

    if len(message) > 38_000:
        message = message[:38_000] + "\n... _(truncated — too many machines to fit in one message)_"
    prefix = f"{cfg.slack_mention} " if cfg.slack_mention else ""
    payload = json.dumps({"text": f"{prefix}{message}"}).encode()
    req = urllib.request.Request(
        cfg.webhook_url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    log.debug("Sending Slack message (%d chars)", len(message))
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status == 200
            if not ok:
                log.warning("Slack returned status %d", resp.status)
            return ok
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        log.error("Slack request failed (%s): %s", type(exc).__name__, exc)
        return False


# ── Display helpers ──────────────────────────────────────────────────────────

def build_status_block(
    machines: list[str],
    details: dict[str, MachineInfo],
    verbose: bool = False,
) -> str:
    lines = ["\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ":computer: *Machine Status:*", ""]
    for m in machines:
        info = details.get(m)
        if info is None:
            lines.append(f":grey_question: `{m}` — *NotFound*")
            continue

        extra = ""
        if info.phase == Phase.PROVISIONED:
            extra = " _(waiting for node)_"
        elif info.phase == Phase.RUNNING and info.node_name:
            ready_tag = ":white_check_mark:" if info.node_ready else ":warning:"
            extra = f" _(node: `{info.node_name}` {ready_tag})_"
        elif info.phase == Phase.FAILED and info.error_message:
            reason = f" [{slack_escape(info.error_reason)}]" if info.error_reason else ""
            extra = f"\n      :warning: _{slack_escape(info.error_message)}{reason}_"

        age_part = f" | age: {info.age_str}" if info.age_str != "n/a" else ""
        instance_part = f" | instance: `{info.instance_id}`" if info.instance_id else ""
        duration_part = f" | provisioned in: {info.provision_duration}" if info.provision_duration else ""

        line = f"  {info.phase.emoji} `{m}`\n      *{info.phase.value}*{extra}{age_part}{instance_part}{duration_part}"
        lines.append(line)

        if verbose and info.last_operation:
            lines.append(f"      _Last op: {slack_escape(info.last_operation.summary())}_")

    return "\n".join(lines)


def build_machineset_block(machinesets: list[MachineSetInfo]) -> str:
    lines = ["\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ":gear: *MachineSet Summary:*", ""]
    for ms in machinesets:
        status_icon = ":white_check_mark:" if ms.ready == ms.desired else ":hourglass_flowing_sand:"
        line = (
            f"  {status_icon} `{ms.name}`\n"
            f"      desired: *{ms.desired}* | replicas: *{ms.replicas}* | "
            f"ready: *{ms.ready}* | available: *{ms.available}*"
        )
        if ms.error_message:
            line += f"\n      :warning: _{slack_escape(ms.error_message)}_"
        lines.append(line)
    return "\n".join(lines)


def print_terminal_table(machines: list[str], details: dict[str, MachineInfo], verbose: bool = False) -> None:
    for m in machines:
        info = details.get(m)
        if info is None:
            log.info("  ❓ %s — NotFound", m)
            continue

        phase_str = colored_phase_str(info.phase)
        extra_parts = []
        if info.node_name:
            ready_str = f"{C.GREEN}Ready{C.RESET}" if info.node_ready else f"{C.YELLOW}NotReady{C.RESET}"
            extra_parts.append(f"node={info.node_name}({ready_str})")
        if info.instance_id:
            extra_parts.append(f"instance={info.instance_id}")
        if info.age_str != "n/a":
            extra_parts.append(f"age={info.age_str}")
        if info.provision_duration:
            extra_parts.append(f"{C.GREEN}provisioned in {info.provision_duration}{C.RESET}")

        extra = f"  {C.DIM}[{', '.join(extra_parts)}]{C.RESET}" if extra_parts else ""
        log.info("  %s %s — %s%s%s%s", phase_str, m, C.BOLD, info.phase.value, C.RESET, extra)

        if info.phase == Phase.FAILED and info.error_message:
            reason = f" [{info.error_reason}]" if info.error_reason else ""
            log.info("      %s⚠️  Error: %s%s%s", C.RED, info.error_message, reason, C.RESET)

        if verbose:
            if info.last_operation:
                log.debug("      %sLast op:%s %s", C.DIM, C.RESET, info.last_operation.summary())
            if info.addresses:
                log.debug("      %sAddresses:%s %s", C.DIM, C.RESET, ", ".join(info.addresses))
            for cond in info.conditions:
                log.debug("      %sCondition:%s %s=%s reason=%s msg=%s", C.DIM, C.RESET, cond.cond_type, cond.status, cond.reason, cond.message)
            for evt in info.events:
                evt_color = C.RED if evt.event_type == "Warning" else C.DIM
                log.debug("      %sEvent:%s %s[%s]%s %s (x%d)", C.DIM, C.RESET, evt_color, evt.reason, C.RESET, evt.message[:100], evt.count)


def print_machineset_summary(machinesets: list[MachineSetInfo]) -> None:
    log.info("%sMachineSet Summary:%s", C.BOLD, C.RESET)
    for ms in machinesets:
        icon = f"{C.GREEN}✅" if ms.ready == ms.desired else f"{C.YELLOW}⏳"
        log.info("  %s%s %s — desired: %d, replicas: %d, ready: %d, available: %d", icon, C.RESET, ms.name, ms.desired, ms.replicas, ms.ready, ms.available)
        if ms.error_message:
            log.info("      %s⚠️  %s%s", C.RED, ms.error_message, C.RESET)


def print_summary(details: dict[str, MachineInfo]) -> None:
    total = len(details)
    by_phase: dict[str, int] = {}
    for info in details.values():
        by_phase[info.phase.value] = by_phase.get(info.phase.value, 0) + 1
    parts = []
    for k, v in sorted(by_phase.items()):
        phase = Phase.from_str(k)
        color = C.GREEN if phase == Phase.RUNNING else C.YELLOW if phase == Phase.PROVISIONING else C.RED if phase == Phase.FAILED else ""
        parts.append(f"{color}{v} {k}{C.RESET}")
    log.info("%sSummary:%s %d machines — %s", C.BOLD, C.RESET, total, ", ".join(parts))


# ── JSON output ──────────────────────────────────────────────────────────────

def _build_status_dict(details: dict[str, MachineInfo], machinesets: list[MachineSetInfo], settlement: Settlement, run: int) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "poll": run,
        "settlement": settlement.value,
        "machinesets": [
            {"name": ms.name, "desired": ms.desired, "replicas": ms.replicas, "ready": ms.ready, "available": ms.available, "error": ms.error_message}
            for ms in machinesets
        ],
        "machines": [
            {
                "name": info.name, "phase": info.phase.value,
                "node": info.node_name, "node_ready": info.node_ready,
                "instance_id": info.instance_id, "age": info.age_str,
                "provision_duration": info.provision_duration,
                "error_message": info.error_message, "error_reason": info.error_reason,
            }
            for info in details.values()
        ],
    }


def emit_json(details: dict[str, MachineInfo], machinesets: list[MachineSetInfo], settlement: Settlement, run: int) -> None:
    print(json.dumps(_build_status_dict(details, machinesets, settlement, run), indent=2), flush=True)


def write_status_file(path: str, details: dict[str, MachineInfo], machinesets: list[MachineSetInfo], settlement: Settlement, run: int) -> None:
    target = Path(path)
    tmp = target.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(_build_status_dict(details, machinesets, settlement, run), indent=2) + "\n")
        tmp.rename(target)
    except OSError as exc:
        log.warning("Failed to write status file %s: %s", path, exc)
        tmp.unlink(missing_ok=True)


# ── Stuck detection ─────────────────────────────────────────────────────────

def check_stuck_machines(
    details: dict[str, MachineInfo],
    phase_timestamps: dict[str, float],
    threshold: int,
    last_stuck_alert: dict[str, float],
) -> list[str]:
    warnings = []
    now = time.time()
    threshold_secs = threshold * 60
    for name, info in details.items():
        if info.phase.is_settled:
            last_stuck_alert.pop(name, None)
            continue
        entered_at = phase_timestamps.get(name, now)
        elapsed = now - entered_at
        if elapsed < threshold_secs:
            continue
        prev_alert = last_stuck_alert.get(name, 0)
        if now - prev_alert < threshold_secs:
            continue
        duration = human_duration(elapsed)
        warnings.append(f"⚠️  `{name}` has been *{info.phase.value}* for *{duration}* (>{threshold}min). May need investigation.")
        log.warning("  %s%s has been %s for %s (>%dm) — may be stuck!%s", C.YELLOW, name, info.phase.value, duration, threshold, C.RESET)
        last_stuck_alert[name] = now
    return warnings


# ── Settlement check ────────────────────────────────────────────────────────

class Settlement(str, Enum):
    ALL_RUNNING = "ALL_RUNNING"
    ALL_FAILED = "ALL_FAILED"
    ALL_DELETING = "ALL_DELETING"
    MIXED_TERMINAL = "MIXED_TERMINAL"
    IN_PROGRESS = "IN_PROGRESS"
    MIN_RUNNING_MET = "MIN_RUNNING_MET"


def check_settlement(details: dict[str, MachineInfo], min_running: int | None = None) -> Settlement:
    running = sum(1 for d in details.values() if d.phase == Phase.RUNNING)
    failed = sum(1 for d in details.values() if d.phase == Phase.FAILED)
    deleting = sum(1 for d in details.values() if d.phase == Phase.DELETING)
    total = len(details)
    settled = sum(1 for d in details.values() if d.phase.is_settled)

    log.debug("Settlement: %d/%d settled (running=%d, failed=%d, deleting=%d)", settled, total, running, failed, deleting)

    if settled < total:
        if min_running is not None and running >= min_running:
            return Settlement.MIN_RUNNING_MET
        return Settlement.IN_PROGRESS
    if deleting == total:
        return Settlement.ALL_DELETING
    if failed == 0 and deleting == 0:
        return Settlement.ALL_RUNNING
    if running == 0 and deleting == 0:
        return Settlement.ALL_FAILED
    return Settlement.MIXED_TERMINAL


# ── CLI argument parsing ────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor OpenShift machines by instance type with Slack notifications.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s -i p5.48xlarge                              # stdout only
  %(prog)s -i p5.48xlarge -i g5.12xlarge               # multiple types
  %(prog)s -m my-machineset-name                       # filter by MachineSet name
  %(prog)s -i p5.48xlarge -s https://hooks.slack...     # stdout + Slack
  %(prog)s -i p5.48xlarge -v -p 15                      # verbose, 15s poll
  %(prog)s -i p5.48xlarge --once --json                 # single snapshot, JSON
  %(prog)s -i p5.48xlarge --timeout 60 --min-running 3  # exit after 60m or 3 running
  %(prog)s -i p5.48xlarge --dry-run -s https://...      # test without sending
  %(prog)s -i p5.48xlarge --heartbeat 10                # alive ping every 10m
  %(prog)s -i p5.48xlarge --lock-machines               # no re-discovery
  %(prog)s -i p5.48xlarge -n custom-namespace           # custom namespace
  SLACK_WEBHOOK_URL=https://... %(prog)s -i p5.48xlarge # env var for Slack

machine phases (openshift/api machine/v1beta1):
  Provisioning → Provisioned → Running (success)
                              → Failed  (unrecoverable error, must delete)
  Any phase    → Deleting    (deletion requested)

exit codes:
  0  All machines running / min-running met / mixed-terminal / user stop
  1  All machines failed / oc login failure
  2  Timeout reached (--timeout)
  3  Still in progress (--once mode)
""",
    )
    filt = parser.add_argument_group("filtering")
    filt.add_argument("-i", "--instance-type", metavar="TYPE", action="append", default=[], help="EC2 instance type (repeatable: -i p5.48xlarge -i g5.12xlarge)")
    filt.add_argument("-m", "--machineset", metavar="NAME", action="append", default=[], help="MachineSet name to monitor directly (repeatable)")
    filt.add_argument("-n", "--namespace", metavar="NS", default=None, help="Namespace for Machine API (default: openshift-machine-api)")
    filt.add_argument("--lock-machines", action="store_true", help="Disable dynamic re-discovery after initial snapshot")

    poll = parser.add_argument_group("polling")
    poll.add_argument("-p", "--poll", metavar="SECONDS", type=int, default=None, help="Poll interval in seconds (default: 30, min: 5)")
    poll.add_argument("--once", action="store_true", help="Run a single snapshot and exit (no polling)")
    poll.add_argument("--timeout", metavar="MINUTES", type=int, default=None, help="Exit with code 2 after this many minutes")

    settle = parser.add_argument_group("settlement")
    settle.add_argument("--min-running", metavar="N", type=int, default=None, help="Exit 0 once at least N machines are Running")
    settle.add_argument("--stuck-threshold", metavar="MINUTES", type=int, default=None, help="Warn after N minutes in a non-terminal phase (default: 15)")

    slack = parser.add_argument_group("slack")
    slack.add_argument("-s", "--slack", metavar="URL", default=None, help="Slack webhook URL (prefer SLACK_WEBHOOK_URL env var for security)")
    slack.add_argument("--slack-mention", metavar="MENTION", default=None, help="Slack mention tag (default: <!channel>). Use <!here>, <@USER_ID>, or empty string")
    slack.add_argument("--dry-run", action="store_true", help="Run full loop but don't actually send Slack messages")
    slack.add_argument("--heartbeat", metavar="MINUTES", type=int, default=None, dest="heartbeat_interval", help="Send a 'still watching' Slack message every N minutes")

    output = parser.add_argument_group("output")
    output.add_argument("-v", "--verbose", action="store_true", help="Debug logging with conditions, events, addresses, lastOperation")
    output.add_argument("-l", "--log-file", metavar="PATH", default=None, help="Also write all log output to this file (appends)")
    output.add_argument("--json", action="store_true", dest="json_output", help="Emit JSON status on each poll to stdout")
    output.add_argument("--no-color", action="store_true", help="Disable ANSI color output")
    output.add_argument("--status-file", metavar="PATH", default=None, help="Write machine status JSON to this file each poll")

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> MonitorConfig:
    if not args.instance_type and not args.machineset:
        print("error: at least one of -i/--instance-type or -m/--machineset is required", file=sys.stderr)
        sys.exit(2)

    webhook_url = args.slack or os.environ.get("SLACK_WEBHOOK_URL") or None
    poll_interval = args.poll if args.poll is not None else int(os.environ.get("POLL_INTERVAL", "30"))
    poll_interval = max(poll_interval, 5)

    if webhook_url and not webhook_url.startswith("https://"):
        print(f"error: Webhook URL must use https://. Got: {webhook_url[:30]}...", file=sys.stderr)
        sys.exit(1)

    if args.timeout is not None and args.timeout < 1:
        print("error: --timeout must be >= 1 minute", file=sys.stderr)
        sys.exit(2)

    return MonitorConfig(
        instance_types=args.instance_type or [],
        machineset_names=args.machineset or [],
        webhook_url=webhook_url,
        poll_interval=poll_interval,
        stuck_threshold_minutes=args.stuck_threshold if args.stuck_threshold is not None else 15,
        verbose=args.verbose,
        json_output=args.json_output,
        once=args.once,
        timeout_minutes=args.timeout,
        namespace=args.namespace or "openshift-machine-api",
        min_running=args.min_running,
        slack_mention=args.slack_mention if args.slack_mention is not None else "<!channel>",
        dry_run=args.dry_run,
        heartbeat_interval_minutes=args.heartbeat_interval,
        lock_machines=args.lock_machines,
        status_file=args.status_file,
        log_file=args.log_file,
        no_color=args.no_color,
    )


# ── Settlement handlers ─────────────────────────────────────────────────────

def handle_settlement(
    settlement: Settlement,
    details: dict[str, MachineInfo],
    machines: list[str],
    cfg: MonitorConfig,
    context: str = "",
) -> int | None:
    status_block = build_status_block(machines, details, cfg.verbose)
    total = len(details)
    filters = cfg.instance_types + cfg.machineset_names
    types_str = ", ".join(f"`{t}`" for t in filters)

    if settlement == Settlement.MIN_RUNNING_MET:
        running = sum(1 for d in details.values() if d.phase == Phase.RUNNING)
        log.info("%s%s%d/%d machines Running — min-running=%d met!%s%s", C.BG_GREEN, C.WHITE, running, total, cfg.min_running, C.RESET, f" {context}" if context else "")
        send_slack(cfg, f":tada: *{running}/{total} machines Running — min-running threshold met!*\n\nFilter: {types_str}\nMonitor complete.{status_block}")
        return 0

    if settlement == Settlement.ALL_RUNNING:
        log.info("%s%sAll %d machines are Running!%s%s", C.BG_GREEN, C.WHITE, total, C.RESET, f" {context}" if context else "")
        send_slack(cfg, f":tada: *All {total} machines are now Running!*\n\nFilter: {types_str}\nMonitor complete.{status_block}")
        return 0

    if settlement == Settlement.ALL_FAILED:
        log.info("%s%sAll %d machines have Failed!%s%s", C.BG_RED, C.WHITE, total, C.RESET, f" {context}" if context else "")
        send_slack(cfg, f":x: *All {total} machines have Failed.*\n\nFilter: {types_str}{status_block}")
        return 1

    if settlement == Settlement.ALL_DELETING:
        log.info("All %d machines are being deleted. Continuing to watch...", total)
        return None

    if settlement == Settlement.MIXED_TERMINAL:
        log.info("%s%sAll machines settled (mixed results).%s%s", C.BG_YELLOW, C.WHITE, C.RESET, f" {context}" if context else "")
        send_slack(cfg, f":warning: *All machines settled (mixed results)*\n\nFilter: {types_str}{status_block}")
        return 0

    return None


# ── Main loop ────────────────────────────────────────────────────────────────

_shutdown_requested = False


def main() -> int:
    global _shutdown_requested

    args = parse_args()
    cfg = build_config(args)
    setup_logging(cfg)

    if args.slack and cfg.webhook_url:
        log.warning("Webhook URL passed via CLI arg — visible in process listings. Prefer SLACK_WEBHOOK_URL env var.")

    phase_timestamps: dict[str, float] = {}
    last_stuck_alert: dict[str, float] = {}
    monitor_start = time.time()
    last_heartbeat = monitor_start

    # ── Pre-flight ───────────────────────────────────────────────────────
    if not check_oc_login():
        return 1

    # ── Banner ───────────────────────────────────────────────────────────
    filters = cfg.instance_types + cfg.machineset_names
    log.info("%s%s", C.BOLD, "=" * 55)
    log.info("OCP Machine Monitor")
    log.info("  Filter(s)        : %s%s%s", C.CYAN, ", ".join(filters), C.RESET)
    log.info("  Namespace        : %s", cfg.namespace)
    log.info("  Poll interval    : %ds", cfg.poll_interval)
    log.info("  Stuck threshold  : %dm", cfg.stuck_threshold_minutes)
    if cfg.timeout_minutes is not None:
        log.info("  Timeout          : %dm", cfg.timeout_minutes)
    if cfg.min_running is not None:
        log.info("  Min running      : %d", cfg.min_running)
    log.info("  Slack            : %s", f"{C.GREEN}enabled{C.RESET}" if cfg.webhook_url else f"{C.DIM}disabled{C.RESET}")
    if cfg.dry_run:
        log.info("  Dry run          : %syes (no Slack sent)%s", C.YELLOW, C.RESET)
    if cfg.heartbeat_interval_minutes:
        log.info("  Heartbeat        : every %dm", cfg.heartbeat_interval_minutes)
    if cfg.lock_machines:
        log.info("  Lock machines    : yes (no re-discovery)")
    if cfg.once:
        log.info("  Mode             : single snapshot (--once)")
    log.info("  Verbose          : %s", cfg.verbose)
    if cfg.json_output:
        log.info("  JSON output      : stdout")
    if cfg.log_file:
        log.info("  Log file         : %s", cfg.log_file)
    if cfg.status_file:
        log.info("  Status file      : %s", cfg.status_file)
    log.info("  First poll       : after 1s")
    log.info("  Press Ctrl+C to stop")
    log.info("=%s%s", "=" * 54, C.RESET)

    # ── Discover ─────────────────────────────────────────────────────────
    discovery = discover(cfg)
    if discovery is None:
        log.error("Discovery failed. Exiting.")
        return 1

    log.info("")
    print_machineset_summary(discovery.machinesets)

    log.info("Found %s%d Machine(s):%s", C.BOLD, len(discovery.machines), C.RESET)
    for m in discovery.machines:
        log.info("  - %s", m)

    # ── Initial snapshot ─────────────────────────────────────────────────
    log.info("")
    log.info("%s── Initial Check ──%s", C.BOLD, C.RESET)

    now = time.time()
    details = parse_machine_items(discovery.machines, discovery.raw_machine_items, cfg)
    if not details:
        log.error("Failed to get machine details. Exiting.")
        return 1

    for name in details:
        phase_timestamps[name] = now

    print_terminal_table(discovery.machines, details, cfg.verbose)
    print_summary(details)

    settlement = check_settlement(details, cfg.min_running)

    if cfg.json_output:
        emit_json(details, discovery.machinesets, settlement, 0)
    if cfg.status_file:
        write_status_file(cfg.status_file, details, discovery.machinesets, settlement, 0)

    if cfg.once:
        exit_code = handle_settlement(settlement, details, discovery.machines, cfg)
        if exit_code is not None:
            return exit_code
        log.info("Machines still in progress (--once mode). Exiting with code 3.")
        return 3

    status_block = build_status_block(discovery.machines, details, cfg.verbose)
    ms_block = build_machineset_block(discovery.machinesets)
    pending = sum(1 for d in details.values() if not d.phase.is_settled)

    exit_code = handle_settlement(settlement, details, discovery.machines, cfg, context="No monitoring needed.")
    if exit_code is not None:
        return exit_code

    log.info("%d/%d settled, %s%d still in progress%s. Starting monitor...", len(details) - pending, len(details), C.YELLOW, pending, C.RESET)
    filters_str = ", ".join(f"`{t}`" for t in filters)
    send_slack(cfg, f":eyes: *OCP Machine Monitor Started*\n\nFilter: {filters_str}\n{pending} of {len(details)} machines still provisioning — polling every {cfg.poll_interval}s{ms_block}{status_block}")

    # ── Signal handler ───────────────────────────────────────────────────
    def shutdown_handler(signum, frame):
        global _shutdown_requested
        _shutdown_requested = True

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # ── Poll loop ────────────────────────────────────────────────────────
    alert_mode = False
    run = 0
    slack_enabled = cfg.webhook_url is not None

    while not _shutdown_requested:
        wait_secs = 1 if run == 0 else cfg.poll_interval
        deadline = time.time() + wait_secs
        while time.time() < deadline and not _shutdown_requested:
            time.sleep(min(1, deadline - time.time()))
        if _shutdown_requested:
            break

        run += 1
        ts = datetime.now().strftime("%H:%M:%S")
        elapsed = human_duration(time.time() - monitor_start)

        # Timeout check
        if cfg.timeout_minutes is not None:
            elapsed_min = (time.time() - monitor_start) / 60
            if elapsed_min >= cfg.timeout_minutes:
                log.warning("%sTimeout reached (%dm). Exiting.%s", C.YELLOW, cfg.timeout_minutes, C.RESET)
                status_block = build_status_block(discovery.machines, details, cfg.verbose)
                send_slack(cfg, f":alarm_clock: *Monitor timed out after {cfg.timeout_minutes}m*\n\n{status_block}")
                return 2

        log.info("")
        log.info("%s── Poll #%d at %s (running %s) ──%s", C.BOLD, run, ts, elapsed, C.RESET)

        # Re-discover (unless locked)
        current_machinesets = discovery.machinesets
        if not cfg.lock_machines:
            new_discovery = discover(cfg)
            if new_discovery is None:
                log.warning("Discovery failed this poll — using previous data. Will retry next poll.")
                send_slack(cfg, f":warning: *Discovery failed* (Poll #{run} at {ts}) — using stale data")
            else:
                current_machinesets = new_discovery.machinesets
                if new_discovery.machines != discovery.machines:
                    added = set(new_discovery.machines) - set(discovery.machines)
                    removed = set(discovery.machines) - set(new_discovery.machines)
                    if added:
                        log.info("%s+ New machines:%s %s", C.GREEN, C.RESET, ", ".join(added))
                        for m in added:
                            phase_timestamps[m] = time.time()
                    if removed:
                        log.info("%s- Machines removed:%s %s", C.RED, C.RESET, ", ".join(removed))
                        for m in removed:
                            phase_timestamps.pop(m, None)
                    discovery = new_discovery
                else:
                    discovery = DiscoveryResult(machinesets=new_discovery.machinesets, machines=discovery.machines, raw_machine_items=new_discovery.raw_machine_items)
        else:
            fresh_ms = oc_get_json("machinesets", cfg)
            if fresh_ms:
                ms_names_set = {ms.name for ms in discovery.machinesets}
                current_machinesets = [
                    _parse_machineset_item(item)
                    for item in fresh_ms.get("items", [])
                    if item["metadata"]["name"] in ms_names_set
                ]
            else:
                log.debug("MachineSet refresh failed (locked mode) — using previous machineset data")
            fresh_data = oc_get_json("machines", cfg)
            if fresh_data:
                locked_set = set(discovery.machines)
                discovery = DiscoveryResult(
                    machinesets=current_machinesets,
                    machines=discovery.machines,
                    raw_machine_items={
                        item["metadata"]["name"]: item
                        for item in fresh_data.get("items", [])
                        if item["metadata"]["name"] in locked_set
                    },
                )
            else:
                log.warning("Machine data refresh failed (locked mode) — using stale data")

        print_machineset_summary(current_machinesets)

        new_details = parse_machine_items(discovery.machines, discovery.raw_machine_items, cfg)
        if not new_details:
            log.warning("Failed to get details. Retrying next poll...")
            continue

        changes: list[str] = []
        change_lines_slack: list[str] = []

        for machine in discovery.machines:
            new_info = new_details.get(machine)
            old_info = details.get(machine)
            new_phase = new_info.phase if new_info else Phase.UNKNOWN
            old_phase = old_info.phase if old_info else Phase.UNKNOWN

            if old_phase != new_phase:
                extra_parts = []
                if new_info and new_info.node_name:
                    ready_str = f"{C.GREEN}Ready{C.RESET}" if new_info.node_ready else f"{C.YELLOW}NotReady{C.RESET}"
                    extra_parts.append(f"node={new_info.node_name}({ready_str})")
                if new_info and new_info.instance_id:
                    extra_parts.append(f"instance={new_info.instance_id}")
                if new_info and new_info.provision_duration:
                    extra_parts.append(f"{C.GREEN}provisioned in {new_info.provision_duration}{C.RESET}")
                extra = f"  {C.DIM}[{', '.join(extra_parts)}]{C.RESET}" if extra_parts else ""

                log.info("  %s %s — %s%s%s  %s<< CHANGED from %s%s%s", colored_phase_str(new_phase), machine, C.BOLD, new_phase.value, C.RESET, C.YELLOW, old_phase.value, C.RESET, extra)

                if new_phase == Phase.FAILED and new_info and new_info.error_message:
                    log.info("      %s⚠️  Error: %s%s", C.RED, new_info.error_message, C.RESET)

                changes.append(f"{machine}: {old_phase.value} -> {new_phase.value}")
                slack_change = f":rotating_light: `{machine}`: *{old_phase.value}* :arrow_right: *{new_phase.value}*"
                if new_phase == Phase.RUNNING and new_info and new_info.provision_duration:
                    slack_change += f" _(provisioned in {new_info.provision_duration})_"
                if new_phase == Phase.FAILED and new_info and new_info.error_message:
                    slack_change += f"\n    :warning: _{slack_escape(new_info.error_message)}_"
                change_lines_slack.append(slack_change)
                phase_timestamps[machine] = time.time()
            else:
                if new_info:
                    extra_parts = []
                    if new_info.node_name:
                        ready_str = f"{C.GREEN}Ready{C.RESET}" if new_info.node_ready else f"{C.YELLOW}NotReady{C.RESET}"
                        extra_parts.append(f"node={new_info.node_name}({ready_str})")
                    if new_info.age_str != "n/a":
                        extra_parts.append(f"age={new_info.age_str}")
                    extra = f"  {C.DIM}[{', '.join(extra_parts)}]{C.RESET}" if extra_parts else ""
                    log.info("  %s %s — %s%s%s%s", colored_phase_str(new_phase), machine, C.BOLD, new_phase.value, C.RESET, extra)

        details = new_details
        print_summary(details)
        settlement = check_settlement(details, cfg.min_running)
        status_block = build_status_block(discovery.machines, details, cfg.verbose)
        ms_block = build_machineset_block(current_machinesets)

        if cfg.json_output:
            emit_json(details, current_machinesets, settlement, run)
        if cfg.status_file:
            write_status_file(cfg.status_file, details, current_machinesets, settlement, run)

        stuck_warnings = check_stuck_machines(details, phase_timestamps, cfg.stuck_threshold_minutes, last_stuck_alert)
        stuck_block = ""
        if stuck_warnings:
            stuck_block = "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n:rotating_light: *Stuck Machines:*\n\n" + "\n".join(stuck_warnings)

        # Heartbeat check
        heartbeat_due = False
        if cfg.heartbeat_interval_minutes:
            if (time.time() - last_heartbeat) / 60 >= cfg.heartbeat_interval_minutes:
                heartbeat_due = True
                last_heartbeat = time.time()

        if changes:
            alert_mode = True
            change_block = "\n".join(change_lines_slack)
            send_slack(cfg, f":rotating_light: *Status Change Detected!* (Poll #{run} at {ts})\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n:arrows_counterclockwise: *Changes:*\n\n{change_block}{status_block}{ms_block}{stuck_block}")
            log.info(">> Phase change detected%s", " — Slack notified" if slack_enabled else "")
        elif alert_mode:
            send_slack(cfg, f":satellite: *Status Update* (Poll #{run} at {ts}){status_block}{ms_block}{stuck_block}")
            log.info(">> Alert mode%s", " — Slack updated" if slack_enabled else "")
        elif stuck_warnings:
            send_slack(cfg, f":warning: *Stuck Machine Warning* (Poll #{run} at {ts}){status_block}{ms_block}{stuck_block}")
            log.info(">> Stuck machine warning sent")
        elif heartbeat_due:
            send_slack(cfg, f":heartbeat: *Monitor heartbeat* (Poll #{run} at {ts}, running {elapsed}){status_block}")
            log.info(">> Heartbeat sent")
        else:
            log.info("No changes.%s", " Slack silent." if slack_enabled else "")

        exit_code = handle_settlement(settlement, details, discovery.machines, cfg, context="Monitor complete.")
        if exit_code is not None:
            return exit_code

        log.info("Next check in %ds...", cfg.poll_interval)

    # ── Graceful shutdown ────────────────────────────────────────────────
    elapsed = human_duration(time.time() - monitor_start)
    log.info("")
    log.info("%s%s", C.BOLD, "=" * 55)
    log.info("Monitor stopped by user after %s", elapsed)
    log.info("Final status:%s", C.RESET)
    print_terminal_table(discovery.machines, details, cfg.verbose)
    print_summary(details)
    log.info("%s%s%s", C.BOLD, "=" * 55, C.RESET)

    status_block = build_status_block(discovery.machines, details, cfg.verbose)
    send_slack(cfg, f":stop_sign: *OCP Machine Monitor stopped by user*\n\nRan for {elapsed}.{status_block}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
