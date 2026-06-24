#!/usr/bin/env python3
"""
deploy-custom-controller-on-operator.py

Deploy / undeploy a custom odh-model-controller on a live RHOAI cluster
using the PVC-based manifest override approach. Mounts your local config/
manifests into the operator pod so templates and params are deployed natively
by the operator (with full kustomize variable substitution).

Works for ANY change to odh-model-controller — new runtimes, params changes,
RBAC updates, webhook config, CRDs, model-serving-api, etc. The operator
deploys whatever is in your local config/ directory.

Based on: https://github.com/opendatahub-io/opendatahub-operator/blob/main/hack/component-dev/README.md

By default, deploy auto-reverts on failure. Pass --no-revert to keep the cluster
in its failed state for debugging.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ────────────────────────────── constants ──────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent

APPS_NS = "redhat-ods-applications"
OPERATOR_NS = "redhat-ods-operator"

NON_INTERACTIVE = False

STATE_DIR = SCRIPT_DIR / ".custom-deploy-state"


def cluster_dir_for(cluster_id: str) -> Path:
    """Return the per-cluster directory for state and logs."""
    return STATE_DIR / cluster_id


def state_file_for(cluster_id: str) -> Path:
    """Return the state file path for a given cluster."""
    return cluster_dir_for(cluster_id) / "state.json"

PVC_NAME = "modelcontroller-manifests"
VOLUME_NAME = "modelcontroller-dev"
MOUNT_PATH = "/opt/manifests/modelcontroller"
CONFIGMAP_NAME = "odh-model-controller-parameters"

EXCLUDE_DIRS = {"samples"}

# ─────────────────────────── colored logging ──────────────────────────

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: DIM,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, RESET)
        tag = record.levelname.ljust(5)
        return f"{color}[{tag}]{RESET} {record.getMessage()}"


log = logging.getLogger("deploy")


def setup_logging(verbose: bool = False) -> None:
    log.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ColorFormatter())
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.addHandler(console)


def add_file_logging(cluster_id: str) -> None:
    """Add a file handler that logs to the cluster's directory."""
    cdir = cluster_dir_for(cluster_id)
    cdir.mkdir(parents=True, exist_ok=True)
    logfile = cdir / "deploy.log"
    if any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == logfile
        for h in log.handlers
    ):
        return
    fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-5s] %(message)s"))
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    log.debug("Log file: %s", logfile)


def header(msg: str) -> None:
    print(f"\n{CYAN}{BOLD}═══ {msg} ═══{RESET}\n", flush=True)


def ok_or_miss(label: str, value: str) -> None:
    if value in ("true", "exists"):
        print(f"    {GREEN}✓{RESET} {label}")
    else:
        print(f"    {YELLOW}✗{RESET} {label} {DIM}({value}){RESET}")


def confirm(prompt: str, default_yes: bool = True) -> bool:
    if NON_INTERACTIVE:
        return True
    suffix = "(Y/n)" if default_yes else "(y/N)"
    try:
        answer = input(f"{prompt} {suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if default_yes:
        return answer.lower() != "n"
    return answer.lower() == "y"


def prompt_input(label: str) -> str:
    try:
        return input(f"{label}> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


# ─────────────────────────── oc() helper ──────────────────────────────

def oc(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    cmd = ["oc", *args]
    log.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=False,
    )
    if capture and result.stdout:
        for line in result.stdout.strip().splitlines():
            log.debug("  stdout: %s", line)
    if capture and result.stderr:
        for line in result.stderr.strip().splitlines():
            log.debug("  stderr: %s", line)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr,
        )
    return result


def run_make(repo_root: Path, target: str, timeout: int = 600, **env_vars: str) -> None:
    make_args = [f"{k}={v}" for k, v in env_vars.items()]
    cmd = ["make", "-C", str(repo_root), target, *make_args]
    log.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.error("make %s timed out after %ds.", target, timeout)
        raise RuntimeError(f"make {target} timed out after {timeout}s")


# ──────────────────────────── prerequisites ───────────────────────────

def require_tools() -> None:
    if not shutil.which("oc"):
        log.error("oc CLI not found. Please install and log in to your cluster.")
        raise SystemExit(1)
    result = subprocess.run(
        ["oc", "whoami"], capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        log.error("Not logged in to an OpenShift cluster. Run 'oc login' first.")
        raise SystemExit(1)


# ──────────────────────────── DeployState ─────────────────────────────

@dataclasses.dataclass
class DeployState:
    cluster_id: str
    csv_name: str
    env_index: int
    original_controller_image: str
    custom_controller_image: str
    original_replicas: int
    original_security_context: dict
    original_num_volumes: int
    original_num_volume_mounts: int
    stock_templates: list[str]
    fsgroup_added: bool
    version: int = 1

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dataclasses.asdict(self)
        data["version"] = 1
        path.write_text(json.dumps(data, indent=2))
        log.info("State saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> DeployState:
        if not path.is_file():
            log.error("No deploy state found at %s.", path)
            log.error("Did you run 'deploy' first?")
            raise SystemExit(1)
        data = json.loads(path.read_text())
        data.pop("original_strategy", None)
        data.pop("original_configmap_image", None)
        data.setdefault("version", 1)
        data.setdefault("fsgroup_added", False)
        try:
            return cls(**data)
        except (TypeError, KeyError) as e:
            log.error("State file %s is corrupt or incompatible: %s", path, e)
            log.error("Delete %s and re-run deploy if stale.", path)
            raise SystemExit(1)


# ──────────────────────────── OcCluster ───────────────────────────────

class OcCluster:
    def __init__(self, operator_ns: str = OPERATOR_NS, apps_ns: str = APPS_NS):
        self.operator_ns = operator_ns
        self.apps_ns = apps_ns
        self._csv_name: str = ""

    def find_csv(self) -> str:
        if self._csv_name:
            return self._csv_name
        result = oc("get", "csv", "-n", self.operator_ns, "-o", "name", check=False)
        for line in result.stdout.strip().splitlines():
            name = line.strip()
            if "rhods-operator" in name.lower() or "rhoai-operator" in name.lower() or "opendatahub-operator" in name.lower():
                csv = name.split("/", 1)[-1]
                log.debug("Matched CSV: %s", csv)
                self._csv_name = csv
                return csv
        return ""

    @property
    def operator_deploy_name(self) -> str:
        csv = self.find_csv()
        if "opendatahub" in csv.lower():
            return "opendatahub-operator"
        return "rhods-operator"

    @property
    def operator_pod_label(self) -> str:
        return f"name={self.operator_deploy_name}"

    def find_env_index(self, csv_name: str, env_name: str) -> int:
        csv = self.get_csv_json(csv_name)
        deploy_spec = csv["spec"]["install"]["spec"]["deployments"][0]["spec"]
        containers = deploy_spec["template"]["spec"]["containers"]
        for c in containers:
            for ei, env in enumerate(c.get("env", [])):
                if env.get("name") == env_name:
                    return ei
        return -1

    def get_csv_env_value(self, csv_name: str, env_index: int) -> str:
        result = oc(
            "get", "csv", csv_name, "-n", self.operator_ns, "-o",
            f"jsonpath={{.spec.install.spec.deployments[0].spec.template.spec.containers[0].env[{env_index}].value}}",
        )
        return result.stdout.strip()

    def get_csv_json(self, csv_name: str) -> dict:
        cmd = ["oc", "get", "csv", csv_name, "-n", self.operator_ns, "-o", "json"]
        log.debug("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        log.debug("CSV JSON: %d bytes (output suppressed)", len(result.stdout))
        return data

    def get_deploy_spec(self, csv_json: dict) -> dict:
        return csv_json["spec"]["install"]["spec"]["deployments"][0]["spec"]

    def wait_for_pvc_bound(self, name: str, ns: str, timeout: int = 120) -> bool:
        """Wait for a PVC to reach Bound state (handles WaitForFirstConsumer)."""
        log.info("Waiting for PVC %s to become Bound (up to %ds)...", name, timeout)
        elapsed = 0
        while elapsed < timeout:
            result = oc(
                "get", "pvc", name, "-n", ns, "-o", "jsonpath={.status.phase}",
                check=False,
            )
            phase = result.stdout.strip()
            if phase == "Bound":
                log.info("PVC %s is Bound.", name)
                return True
            log.debug("PVC %s phase: %s (elapsed %ds)", name, phase or "Pending", elapsed)
            time.sleep(5)
            elapsed += 5
        log.warning("PVC %s still %s after %ds.", name, phase or "Pending", timeout)
        return False

    def wait_for_rollout(self, deploy: str, ns: str, timeout: str = "120s") -> None:
        log.info("Waiting for %s rollout in %s...", deploy, ns)
        oc("rollout", "status", f"deployment/{deploy}", "-n", ns, f"--timeout={timeout}")

    def wait_for_pod_ready(self, ns: str, label: str, timeout: int = 120) -> bool:
        log.info("Waiting for pod (%s) to be ready...", label)
        elapsed = 0
        while elapsed < timeout:
            result = oc(
                "get", "pods", "-n", ns, "-l", label, "-o",
                "jsonpath={.items[0].status.phase}", check=False,
            )
            phase = result.stdout.strip()
            if phase == "Running":
                result = oc(
                    "get", "pods", "-n", ns, "-l", label, "-o",
                    'jsonpath={.items[0].status.conditions[?(@.type=="Ready")].status}',
                    check=False,
                )
                if result.stdout.strip() == "True":
                    return True
            time.sleep(2)
            elapsed += 2
        log.warning("Pod did not become ready within %ds.", timeout)
        return False

    def get_operator_pod(self) -> str:
        result = oc(
            "get", "pods", "-n", self.operator_ns, "-l", self.operator_pod_label,
            "-o", "jsonpath={.items[0].metadata.name}", check=False,
        )
        return result.stdout.strip()

    def resource_exists(self, *args: str) -> bool:
        result = oc("get", *args, check=False)
        return result.returncode == 0

    def get_templates(self, ns: str) -> list[str]:
        result = oc(
            "get", "templates", "-n", ns, "-o",
            r"jsonpath={range .items[*]}{.metadata.name}{'\n'}{end}",
            check=False,
        )
        return [t for t in result.stdout.strip().splitlines() if t.strip()]

    def get_deployment_image(self, name: str, ns: str) -> str:
        result = oc(
            "get", "deployment", name, "-n", ns, "-o",
            "jsonpath={.spec.template.spec.containers[0].image}", check=False,
        )
        return result.stdout.strip() or "N/A"

    def get_cluster_id(self) -> str:
        """Return a stable, human-readable cluster identifier.

        Prefers the OpenShift infrastructure name; falls back to a short hash
        of the API server URL for non-OpenShift or minimal clusters.
        """
        result = oc(
            "get", "infrastructure", "cluster", "-o",
            "jsonpath={.status.infrastructureName}", check=False,
        )
        name = result.stdout.strip()
        if name:
            return name
        result = oc("whoami", "--show-server", check=False)
        url = result.stdout.strip()
        return hashlib.sha256(url.encode()).hexdigest()[:12]


# ───────────────────────────── deploy ─────────────────────────────────

def _resolve_config_dir() -> Path:
    """Auto-detect the odh-model-controller config/ directory, or prompt."""
    candidate = SCRIPT_DIR.parent / "config"
    if candidate.is_dir():
        return candidate
    sibling = SCRIPT_DIR.parent.parent / "odh-model-controller" / "config"
    if sibling.is_dir():
        return sibling
    print(f"{CYAN}Could not auto-detect config/ directory.{RESET}")
    print("  Enter the path to the odh-model-controller repo (must contain config/):")
    raw = prompt_input("")
    if not raw:
        log.error("Config directory is required.")
        raise SystemExit(1)
    p = Path(raw).expanduser().resolve() / "config"
    if not p.is_dir():
        log.error("config/ not found at %s", p)
        raise SystemExit(1)
    return p


def cmd_deploy(
    args: argparse.Namespace,
    *,
    controller_image: str | None = None,
    manifest_src: Path | None = None,
    server_image: str | None = None,
) -> None:
    cluster = OcCluster(operator_ns=args.operator_ns, apps_ns=args.apps_ns)

    img = controller_image or getattr(args, "controller_image", None) or ""
    if not img:
        print(f"{CYAN}Enter the custom odh-model-controller image:{RESET}")
        print("  Example: quay.io/myuser/odh-model-controller:my-branch")
        img = prompt_input("")
        if not img:
            log.error("Controller image is required.")
            raise SystemExit(1)

    src = manifest_src or _resolve_config_dir()
    no_revert = getattr(args, "no_revert", False)

    cluster_id = cluster.get_cluster_id()
    add_file_logging(cluster_id)
    sf = state_file_for(cluster_id)

    if sf.is_file():
        log.warning("A previous deployment state exists for cluster %s.", cluster_id)
        print(f"{YELLOW}Run 'undeploy' first, or delete {sf} if stale.{RESET}")
        if not confirm("Continue anyway?", default_yes=False):
            raise SystemExit(0)

    header("Deploying custom odh-model-controller (PVC mount)")
    print(f"  Cluster          : {BOLD}{cluster_id}{RESET}")
    print(f"  Controller image : {BOLD}{img}{RESET}")
    print(f"  Manifests from   : {BOLD}{src}{RESET}")
    print()
    if not confirm("Proceed?"):
        log.info("Aborted.")
        raise SystemExit(0)

    # ── Step 1: Discover CSV and capture original state ──
    header("Step 1/6 — Discovering operator CSV and capturing state")

    csv_name = cluster.find_csv()
    if not csv_name:
        log.error("Could not find RHOAI operator CSV in %s.", cluster.operator_ns)
        raise SystemExit(1)
    log.info("Found CSV: %s", csv_name)

    env_index = cluster.find_env_index(csv_name, "RELATED_IMAGE_ODH_MODEL_CONTROLLER_IMAGE")
    if env_index == -1:
        log.error("Could not find RELATED_IMAGE_ODH_MODEL_CONTROLLER_IMAGE in CSV.")
        raise SystemExit(1)
    log.info("Env var index: %d", env_index)

    original_image = cluster.get_csv_env_value(csv_name, env_index)
    log.info("Original image: %s", original_image)

    csv_json = cluster.get_csv_json(csv_name)
    deploy_spec = cluster.get_deploy_spec(csv_json)
    pod_spec = deploy_spec["template"]["spec"]

    original_replicas = deploy_spec.get("replicas", 3)
    original_security_context = pod_spec.get("securityContext", {})
    original_num_volumes = len(pod_spec.get("volumes", []))
    original_num_volume_mounts = len(pod_spec["containers"][0].get("volumeMounts", []))

    log.info("Original replicas: %d", original_replicas)
    log.info("Original volumes: %d, volumeMounts: %d", original_num_volumes, original_num_volume_mounts)
    log.debug(
        "Parsed deployment spec: replicas=%d, volumes=%d, volumeMounts=%d",
        original_replicas, original_num_volumes, original_num_volume_mounts,
    )

    stock_templates = cluster.get_templates(cluster.apps_ns)
    log.info("Stock templates captured: %d", len(stock_templates))

    state = DeployState(
        cluster_id=cluster_id,
        csv_name=csv_name,
        env_index=env_index,
        original_controller_image=original_image,
        custom_controller_image=img,
        original_replicas=original_replicas,
        original_security_context=original_security_context,
        original_num_volumes=original_num_volumes,
        original_num_volume_mounts=original_num_volume_mounts,
        stock_templates=stock_templates,
        fsgroup_added=False,
    )
    state.save(sf)

    if no_revert:
        log.info("Auto-revert on failure: %sdisabled%s (run 'undeploy' to restore manually)", BOLD, RESET)
    else:
        log.info("Auto-revert on failure: %senabled%s (use --no-revert to disable)", BOLD, RESET)

    try:
        _deploy_steps_2_to_6(cluster, state, img, src, csv_name, env_index, server_image=server_image)
    except (Exception, KeyboardInterrupt) as exc:
        if no_revert:
            log.error("Deploy failed: %s", exc)
            log.error("--no-revert set. Run 'undeploy' to restore manually.")
            raise SystemExit(1)
        print()
        log.error("Deploy failed: %s", exc)
        if hasattr(exc, "stderr") and exc.stderr:
            for line in exc.stderr.strip().splitlines():
                log.error("  %s", line)
        log.error("═══════════════════════════════════════════════════════")
        log.error(" Auto-reverting cluster to stock state")
        log.error("═══════════════════════════════════════════════════════")
        try:
            cmd_undeploy(
                argparse.Namespace(operator_ns=cluster.operator_ns, apps_ns=cluster.apps_ns),
                interactive=False, cluster_id_override=cluster_id,
            )
            log.info("Auto-revert complete. Cluster restored to stock state.")
        except Exception as revert_err:
            log.error("Auto-revert FAILED: %s. Run 'undeploy' manually.", revert_err)
        raise SystemExit(1)

    # ── Summary ──
    header("Deploy complete")
    print(f"  {GREEN}✓{RESET} PVC created               → {BOLD}{PVC_NAME}{RESET} in {cluster.operator_ns}")
    print(f"  {GREEN}✓{RESET} CSV patched                → {BOLD}{csv_name}{RESET}")
    print(f"  {GREEN}✓{RESET} Manifests copied           → {BOLD}{MOUNT_PATH}{RESET}")
    print(f"  {GREEN}✓{RESET} Controller image           → {BOLD}{img}{RESET}")
    print()

    print(f"  {BOLD}Runtime templates:{RESET}")
    result = oc(
        "get", "templates", "-n", cluster.apps_ns,
        "-l", "app.kubernetes.io/part-of in (odh-dashboard, odh-model-controller)",
        "--no-headers", check=False,
    )
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            print(f"    {parts[0]:<45s} {parts[1] if len(parts) > 1 else ''}")
    else:
        print("    (none found)")
    print()

    _print_pods("Controller pod", cluster.apps_ns, "app=odh-model-controller")
    _print_pods("Model-serving-api pod", cluster.apps_ns, "app.kubernetes.io/name=model-serving-api")
    print()
    log.info("To undo all changes: %sundeploy%s", BOLD, RESET)
    log.info("To see full cluster state: %sstatus%s", BOLD, RESET)


def _deploy_steps_2_to_6(
    cluster: OcCluster,
    state: DeployState,
    controller_image: str,
    manifest_src: Path,
    csv_name: str,
    env_index: int,
    server_image: str | None = None,
) -> None:
    OPERATOR_NS = cluster.operator_ns  # noqa: N806
    APPS_NS = cluster.apps_ns  # noqa: N806
    # ── Step 2: Create PVC ──
    header("Step 2/6 — Creating PVC for manifests")

    if cluster.resource_exists("pvc", PVC_NAME, "-n", OPERATOR_NS):
        log.info("PVC %s already exists — reusing.", PVC_NAME)
    else:
        pvc_yaml = f"""\
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: {PVC_NAME}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
  volumeMode: Filesystem
"""
        proc = subprocess.run(
            ["oc", "create", "-n", OPERATOR_NS, "-f", "-"],
            input=pvc_yaml, text=True, check=True, capture_output=True,
        )
        log.debug("  stdout: %s", proc.stdout.strip())
        log.info("PVC %s created.", PVC_NAME)

    # ── Step 3: Patch CSV ──
    header("Step 3/6 — Patching CSV")

    # Query the namespace's allocated supplemental-groups range and use the first
    # GID as fsGroup. This ensures the PVC mount is group-writable by the pod UID
    # and is accepted by restricted-v2 SCC (which requires fsGroup from the range).
    fs_group = None
    sg_result = oc(
        "get", "namespace", OPERATOR_NS, "-o",
        "jsonpath={.metadata.annotations.openshift\\.io/sa\\.scc\\.supplemental-groups}",
        check=False,
    )
    sg_range = sg_result.stdout.strip()
    if sg_range:
        fs_group = int(sg_range.split("/")[0])
        log.info("Namespace supplemental-groups range: %s → using fsGroup=%d", sg_range, fs_group)
    else:
        log.warning("Could not determine namespace supplemental-groups range; skipping fsGroup patch")

    # Don't override the deployment strategy — keeping RollingUpdate avoids
    # a deadlock with WaitForFirstConsumer storage classes (AWS EBS gp3-csi, etc.)
    # where Recreate kills all pods before the PVC can bind.
    patch_ops: list[dict[str, Any]] = [
        {"op": "replace", "path": "/spec/install/spec/deployments/0/spec/replicas", "value": 1},
        {"op": "add", "path": "/spec/install/spec/deployments/0/spec/template/spec/containers/0/volumeMounts/-",
         "value": {"name": VOLUME_NAME, "mountPath": MOUNT_PATH}},
        {"op": "add", "path": "/spec/install/spec/deployments/0/spec/template/spec/volumes/-",
         "value": {"name": VOLUME_NAME, "persistentVolumeClaim": {"claimName": PVC_NAME}}},
        {"op": "replace",
         "path": f"/spec/install/spec/deployments/0/spec/template/spec/containers/0/env/{env_index}/value",
         "value": controller_image},
    ]
    if fs_group is not None:
        # Check if securityContext already exists on the pod spec
        sc_result = oc(
            "get", "csv", csv_name, "-n", OPERATOR_NS, "-o",
            "jsonpath={.spec.install.spec.deployments[0].spec.template.spec.securityContext}",
            check=False,
        )
        if sc_result.stdout.strip():
            patch_ops.append({
                "op": "add",
                "path": "/spec/install/spec/deployments/0/spec/template/spec/securityContext/fsGroup",
                "value": fs_group,
            })
        else:
            patch_ops.append({
                "op": "add",
                "path": "/spec/install/spec/deployments/0/spec/template/spec/securityContext",
                "value": {"fsGroup": fs_group},
            })

    csv_patch = json.dumps(patch_ops)
    log.debug("JSON patch: %s", csv_patch)

    oc("patch", "csv", csv_name, "-n", OPERATOR_NS, "--type=json", "-p", csv_patch)

    if fs_group is not None:
        state.fsgroup_added = True
        state.save(state_file_for(state.cluster_id))

    log.info("CSV patched:")
    log.info("  - replicas: 1")
    log.info("  - strategy: kept original (avoids WaitForFirstConsumer deadlock)")
    if fs_group is not None:
        log.info("  - fsGroup: %d (from namespace supplemental-groups range)", fs_group)
    log.info("  - volume: %s -> PVC %s", VOLUME_NAME, PVC_NAME)
    log.info("  - volumeMount: %s", MOUNT_PATH)
    log.info("  - controller image: %s", controller_image)

    # ── Step 4: Wait for operator pod with PVC ──
    header("Step 4/6 — Waiting for operator pod with PVC mount")

    # WaitForFirstConsumer storage classes (e.g. AWS EBS gp3-csi) keep the PVC
    # Pending until the pod referencing it is scheduled.  Give the full chain
    # (OLM reconcile -> pod schedule -> volume provision -> attach) time.
    if not cluster.wait_for_pvc_bound(PVC_NAME, OPERATOR_NS, timeout=180):
        raise RuntimeError(f"PVC {PVC_NAME} did not bind within 180s.")
    cluster.wait_for_rollout(cluster.operator_deploy_name, OPERATOR_NS, "300s")
    if not cluster.wait_for_pod_ready(OPERATOR_NS, cluster.operator_pod_label, 180):
        raise RuntimeError("Operator pod not ready after 180s.")

    operator_pod = cluster.get_operator_pod()
    if not operator_pod:
        raise RuntimeError("Could not find operator pod.")
    log.info("Operator pod: %s", operator_pod)

    result = oc(
        "exec", "-n", OPERATOR_NS, operator_pod, "--", "ls", MOUNT_PATH, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"PVC mount at {MOUNT_PATH} not accessible in operator pod.")
    log.info("PVC mount verified at %s.", MOUNT_PATH)

    # ── Step 5: Copy manifests ──
    header("Step 5/6 — Copying manifests into operator pod")
    log.info("Copying %s/ -> %s:%s/", manifest_src, operator_pod, MOUNT_PATH)

    dirs_to_copy = sorted(
        d.name for d in manifest_src.iterdir()
        if d.is_dir() and d.name not in EXCLUDE_DIRS
    )
    log.debug("Directories to copy: %s", dirs_to_copy)

    for dirname in dirs_to_copy:
        src_dir = manifest_src / dirname
        oc("cp", str(src_dir), f"{OPERATOR_NS}/{operator_pod}:{MOUNT_PATH}/{dirname}")
        log.info("  copied: %s/", dirname)

    metadata_file = manifest_src / "component_metadata.yaml"
    if metadata_file.is_file():
        oc("cp", str(metadata_file), f"{OPERATOR_NS}/{operator_pod}:{MOUNT_PATH}/component_metadata.yaml")
        log.info("  copied: component_metadata.yaml")

    log.info("Verifying all copied directories on PVC...")
    verify_ok = True
    for dirname in dirs_to_copy:
        result = oc(
            "exec", "-n", OPERATOR_NS, operator_pod, "--", "test", "-d", f"{MOUNT_PATH}/{dirname}",
            check=False,
        )
        if result.returncode == 0:
            log.info("    %s✓%s %s/", GREEN, RESET, dirname)
        else:
            log.warning("    ✗ %s/ — missing on PVC", dirname)
            verify_ok = False
    if not verify_ok:
        raise RuntimeError("Manifest copy verification failed — aborting.")

    params_env = f"{MOUNT_PATH}/base/params.env"
    log.info("Patching params.env on PVC to use custom controller image...")
    oc(
        "exec", "-n", OPERATOR_NS, operator_pod, "--",
        "sed", "-i",
        f"s|^odh-model-controller=.*|odh-model-controller={controller_image}|",
        params_env,
    )
    result = oc(
        "exec", "-n", OPERATOR_NS, operator_pod, "--",
        "grep", "^odh-model-controller=", params_env,
        check=False,
    )
    log.info("  params.env: %s", result.stdout.strip())

    if server_image:
        log.info("Patching params.env on PVC to use custom server image...")
        oc(
            "exec", "-n", OPERATOR_NS, operator_pod, "--",
            "sed", "-i",
            f"s|^odh-model-serving-api=.*|odh-model-serving-api={server_image}|",
            params_env,
        )
        result = oc(
            "exec", "-n", OPERATOR_NS, operator_pod, "--",
            "grep", "^odh-model-serving-api=", params_env,
            check=False,
        )
        log.info("  params.env: %s", result.stdout.strip())

    # ── Step 6: Restart operator and patch configmap ──
    header("Step 6/6 — Restarting operator to apply manifests")

    # Delete the operator pod directly instead of `oc rollout restart`.
    # OLM continuously reconciles the Deployment from the CSV and reverts
    # the `restartedAt` annotation that `rollout restart` adds, so the pod
    # never actually restarts. Deleting the pod lets the ReplicaSet
    # controller recreate it without OLM interference.
    log.info("Deleting operator pod to force restart (OLM-safe)...")
    oc("delete", "pod", "-n", OPERATOR_NS, "-l", cluster.operator_pod_label)
    if not cluster.wait_for_pod_ready(OPERATOR_NS, cluster.operator_pod_label, 180):
        raise RuntimeError("Operator pod not ready after restart.")

    log.info("Waiting for controller image to propagate...")
    current_image = ""
    for _ in range(30):
        current_image = cluster.get_deployment_image("odh-model-controller", APPS_NS)
        if current_image == controller_image:
            break
        time.sleep(5)

    if current_image != controller_image:
        log.warning("╔══════════════════════════════════════════════════════════════╗")
        log.warning("║  IMAGE PROPAGATION INCOMPLETE                               ║")
        log.warning("╚══════════════════════════════════════════════════════════════╝")
        log.warning(
            "Operator did not propagate image within 150s (current: %s).",
            current_image,
        )
        log.warning("Deploy applied (PVC, CSV, manifests) but the operator's reconciliation")
        log.warning("loop hasn't updated the Deployment image yet. Run 'status' to check later.")
        raise SystemExit(2)
    else:
        log.info("Controller deployment updated with custom image.")
        try:
            cluster.wait_for_rollout("odh-model-controller", APPS_NS, "120s")
        except subprocess.CalledProcessError:
            log.warning("Controller rollout did not complete within timeout.")
            result = oc(
                "get", "pods", "-n", APPS_NS, "-l", "app=odh-model-controller",
                "-o", "custom-columns=NAME:.metadata.name,STATUS:.status.phase,REASON:.status.containerStatuses[0].state.waiting.reason",
                "--no-headers", check=False,
            )
            if result.stdout.strip():
                log.warning("  Pod status:\n%s", result.stdout.strip())


# ──────────────────────────── build-deploy ────────────────────────────

def cmd_build_deploy(args: argparse.Namespace) -> None:
    img: str = getattr(args, "image", "") or ""
    tag: str = getattr(args, "tag", "") or ""
    repo_root_arg: str = getattr(args, "repo_root", "") or ""
    with_server: bool = getattr(args, "with_server", False)

    # Resolve repo root
    if repo_root_arg:
        repo_root = Path(repo_root_arg).expanduser().resolve()
    elif (SCRIPT_DIR.parent / "Makefile").is_file() and (SCRIPT_DIR.parent / "config").is_dir():
        repo_root = SCRIPT_DIR.parent
    elif (SCRIPT_DIR.parent.parent / "odh-model-controller" / "Makefile").is_file():
        repo_root = SCRIPT_DIR.parent.parent / "odh-model-controller"
    else:
        print(f"{CYAN}Could not auto-detect the odh-model-controller repo root.{RESET}")
        print("  Enter the path to the repo (must contain Makefile and config/):")
        raw = prompt_input("")
        if not raw:
            log.error("Repo root is required.")
            raise SystemExit(1)
        repo_root = Path(raw).expanduser().resolve()

    if not (repo_root / "Makefile").is_file():
        log.error("Makefile not found in %s. Is this the odh-model-controller repo?", repo_root)
        raise SystemExit(1)
    if not (repo_root / "config").is_dir():
        log.error("config/ directory not found in %s.", repo_root)
        raise SystemExit(1)

    manifest_src = repo_root / "config"

    if img and tag:
        log.warning("--image and --tag are mutually exclusive. Using --image.")
        tag = ""

    user = os.environ.get("USER", "user")
    if not img:
        default_repo = f"quay.io/{user}/odh-model-controller"
        if tag:
            img = f"{default_repo}:{tag}"
        else:
            try:
                branch = subprocess.run(
                    ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True, check=True,
                ).stdout.strip().replace("/", "-")
            except (subprocess.CalledProcessError, FileNotFoundError):
                branch = "dev"
            img = f"{default_repo}:{branch}"
            print(f"{CYAN}No --tag or --image specified. Using:{RESET}")
            print(f"  {BOLD}{img}{RESET}")
            print()
            if not confirm("OK?"):
                log.info("Aborted.")
                raise SystemExit(0)

    header("Building and pushing container image")
    print(f"  Repo root:    {BOLD}{repo_root}{RESET}")
    print(f"  Image:        {BOLD}{img}{RESET}")
    print(f"  With server:  {BOLD}{with_server}{RESET}")
    print()

    run_make(repo_root, "container-build", IMG=img)
    print()
    run_make(repo_root, "container-push", IMG=img)

    if with_server:
        if "@" in img:
            log.error("Digest-based images are not supported with --with-server; use a tag-based image.")
            raise SystemExit(1)
        basename = img.rsplit("/", 1)[-1]
        server_tag = basename.rsplit(":", 1)[-1] if ":" in basename else "latest"
        server_img = f"quay.io/{user}/odh-model-serving-api:{server_tag}"

        print()
        run_make(repo_root, "container-build-server", SERVER_IMG=server_img)
        print()
        run_make(repo_root, "container-push-server", SERVER_IMG=server_img)

        log.info("Server image built and pushed: %s", server_img)

    header("Build + push complete — starting deploy")
    cmd_deploy(
        args,
        controller_image=img,
        manifest_src=manifest_src,
        server_image=server_img if with_server else None,
    )


# ──────────────────────────── undeploy ────────────────────────────────

def cmd_undeploy(
    args: argparse.Namespace,
    interactive: bool = True,
    cluster_id_override: str | None = None,
) -> None:
    cluster = OcCluster(
        operator_ns=getattr(args, "operator_ns", OPERATOR_NS),
        apps_ns=getattr(args, "apps_ns", APPS_NS),
    )
    OPERATOR_NS_L = cluster.operator_ns
    APPS_NS_L = cluster.apps_ns

    header("Undeploying custom odh-model-controller")

    cid = cluster_id_override or cluster.get_cluster_id()
    add_file_logging(cid)
    sf = state_file_for(cid)
    state = DeployState.load(sf)

    if state.cluster_id != "unknown" and state.cluster_id != cid:
        log.error(
            "State file is for cluster %s but you are logged into %s.",
            state.cluster_id, cid,
        )
        log.error("Switch to the correct cluster or delete %s if stale.", sf)
        raise SystemExit(1)

    print(f"  Cluster            : {BOLD}{cid}{RESET}")
    print(f"  CSV                : {BOLD}{state.csv_name}{RESET}")
    print(f"  Restore image      : {BOLD}{state.original_controller_image}{RESET}")
    print(f"  Restore replicas   : {BOLD}{state.original_replicas}{RESET}")
    print()

    if interactive:
        if not confirm("Proceed with undeploy?"):
            log.info("Aborted.")
            raise SystemExit(0)

    # ── Step 1: Restore CSV ──
    header("Step 1/3 — Restoring CSV to original state")

    csv_needs_restore = False
    try:
        csv_json = cluster.get_csv_json(state.csv_name)
        deploy_spec = cluster.get_deploy_spec(csv_json)
        pod_spec = deploy_spec["template"]["spec"]
        vols = pod_spec.get("volumes", [])
        csv_needs_restore = any(v.get("name") == VOLUME_NAME for v in vols)
    except Exception as e:
        log.warning("Could not inspect CSV: %s — will attempt restore anyway.", e)
        csv_needs_restore = True

    if csv_needs_restore:
        # Derive volume/volumeMount indices dynamically from the live CSV
        # rather than using stored counts, to handle OLM reconcile reordering.
        vol_idx = None
        vm_idx = None
        try:
            for i, v in enumerate(vols):
                if v.get("name") == VOLUME_NAME:
                    vol_idx = str(i)
                    break
            containers = pod_spec.get("containers", [{}])
            for i, vm in enumerate(containers[0].get("volumeMounts", [])):
                if vm.get("name") == VOLUME_NAME:
                    vm_idx = str(i)
                    break
        except (IndexError, KeyError, NameError):
            pass

        if vol_idx is None:
            vol_idx = str(state.original_num_volumes)
            log.debug("Volume %s not found by name; falling back to stored index %s", VOLUME_NAME, vol_idx)
        if vm_idx is None:
            vm_idx = str(state.original_num_volume_mounts)
            log.debug("VolumeMount %s not found by name; falling back to stored index %s", VOLUME_NAME, vm_idx)

        ops: list[dict[str, Any]] = [
            {"op": "remove", "path": f"/spec/install/spec/deployments/0/spec/template/spec/volumes/{vol_idx}"},
            {"op": "remove", "path": f"/spec/install/spec/deployments/0/spec/template/spec/containers/0/volumeMounts/{vm_idx}"},
            {"op": "replace", "path": "/spec/install/spec/deployments/0/spec/replicas", "value": state.original_replicas},
            {"op": "replace",
             "path": f"/spec/install/spec/deployments/0/spec/template/spec/containers/0/env/{state.env_index}/value",
             "value": state.original_controller_image},
        ]
        if state.fsgroup_added:
            if state.original_security_context:
                ops.append({
                    "op": "replace",
                    "path": "/spec/install/spec/deployments/0/spec/template/spec/securityContext",
                    "value": state.original_security_context,
                })
            else:
                ops.append({
                    "op": "remove",
                    "path": "/spec/install/spec/deployments/0/spec/template/spec/securityContext",
                })

        restore_patch = json.dumps(ops)
        log.debug("Restore patch: %s", restore_patch)

        try:
            oc("patch", "csv", state.csv_name, "-n", OPERATOR_NS_L, "--type=json", "-p", restore_patch)
            log.info("CSV restored:")
            log.info("  - replicas: %d", state.original_replicas)
            log.info("  - volume & volumeMount: removed")
            log.info("  - controller image: restored")

            # OLM will reconcile the Deployment from the restored CSV.
            # Delete the operator pod to force a clean restart so the
            # operator re-reads the original RELATED_IMAGE env vars and
            # re-renders the kustomize manifests with stock images.
            log.info("Deleting operator pod to force restart with restored CSV...")
            oc("delete", "pod", "-n", OPERATOR_NS_L, "-l", cluster.operator_pod_label, check=False)
            if not cluster.wait_for_pod_ready(OPERATOR_NS_L, cluster.operator_pod_label, 180):
                log.warning("Operator pod not ready within 180s.")

            log.info("Waiting for controller image to revert...")
            current_image = ""
            for _ in range(40):
                current_image = cluster.get_deployment_image("odh-model-controller", APPS_NS_L)
                if current_image == state.original_controller_image:
                    break
                time.sleep(5)

            if current_image == state.original_controller_image:
                log.info("Controller image reverted.")
            else:
                log.warning("Controller image may not have reverted yet (current: %s).", current_image)
                log.warning("The operator should eventually reconcile it back.")

            try:
                cluster.wait_for_rollout("odh-model-controller", APPS_NS_L, "120s")
            except subprocess.CalledProcessError:
                pass

        except Exception as e:
            log.error("CSV restore patch failed: %s", e)
            log.error("The cluster may need manual CSV cleanup.")
    else:
        log.info("CSV was not patched (dev volume not present) — no CSV restore needed.")

    # ── Step 2: Clean up extra templates + Delete PVC ──
    header("Step 2/3 — Cleaning up extra templates and PVC")

    stock_set = set(state.stock_templates)
    current_templates = cluster.get_templates(APPS_NS_L)

    for tmpl in current_templates:
        if tmpl not in stock_set:
            result = oc("delete", "template", tmpl, "-n", APPS_NS_L, check=False)
            if result.returncode == 0:
                log.info("Removed extra template: %s", tmpl)
    log.info("Template cleanup done.")

    if cluster.resource_exists("pvc", PVC_NAME, "-n", OPERATOR_NS_L):
        oc("delete", "pvc", PVC_NAME, "-n", OPERATOR_NS_L, "--wait=false")
        log.info("PVC %s deletion initiated.", PVC_NAME)
    else:
        log.info("PVC %s already absent.", PVC_NAME)

    # ── Step 3: Clean up state ──
    header("Step 3/3 — Cleaning up state")
    cdir = cluster_dir_for(cid)
    log.info("Removing cluster directory: %s", cdir)
    for fh in log.handlers[:]:
        if isinstance(fh, logging.FileHandler) and Path(fh.baseFilename).parent == cdir:
            log.removeHandler(fh)
            fh.close()
    shutil.rmtree(cdir, ignore_errors=True)
    log.info("Cluster state cleaned up (%s).", cid)

    # ── Summary ──
    if interactive:
        header("Undeploy complete — cluster restored")
        print(f"  {GREEN}✓{RESET} CSV restored (replicas, volumes, image)")
        print(f"  {GREEN}✓{RESET} Extra templates cleaned (diff-based)")
        print(f"  {GREEN}✓{RESET} PVC deleted")
        print(f"  {GREEN}✓{RESET} State file cleaned up")
        print()
        _print_pods("Controller pod", APPS_NS_L, "app=odh-model-controller")


# ──────────────────────────── status ──────────────────────────────────

def cmd_status(args: argparse.Namespace) -> None:
    cluster = OcCluster(
        operator_ns=getattr(args, "operator_ns", OPERATOR_NS),
        apps_ns=getattr(args, "apps_ns", APPS_NS),
    )
    OPERATOR_NS_L = cluster.operator_ns  # noqa: N806
    APPS_NS_L = cluster.apps_ns  # noqa: N806
    cid = cluster.get_cluster_id()
    cdir = cluster_dir_for(cid)
    if cdir.is_dir():
        add_file_logging(cid)

    header(f"Current cluster state ({cid})")

    csv_name = cluster.find_csv()

    csv_image = "N/A"
    replicas: Any = "?"
    has_dev_volume = False

    if not csv_name:
        log.warning("No RHOAI operator CSV found in %s — CSV section unavailable.", OPERATOR_NS_L)
    else:
        env_index = cluster.find_env_index(csv_name, "RELATED_IMAGE_ODH_MODEL_CONTROLLER_IMAGE")
        if env_index != -1:
            csv_image = cluster.get_csv_env_value(csv_name, env_index)
        else:
            csv_image = "(env var not found)"
        result = oc(
            "get", "csv", csv_name, "-n", OPERATOR_NS_L, "-o",
            "jsonpath={.spec.install.spec.deployments[0].spec.replicas}", check=False,
        )
        replicas = result.stdout.strip() or "?"

        csv_json = cluster.get_csv_json(csv_name)
        deploy_spec = cluster.get_deploy_spec(csv_json)
        vols = deploy_spec["template"]["spec"].get("volumes", [])
        has_dev_volume = any(v.get("name") == VOLUME_NAME for v in vols)

    deploy_image = cluster.get_deployment_image("odh-model-controller", APPS_NS_L)
    pvc_exists = cluster.resource_exists("pvc", PVC_NAME, "-n", OPERATOR_NS_L)

    # CSV & Controller Deployment
    print(f"  {BOLD}CSV{RESET} ({csv_name or 'none'}):")
    print(f"    controller image = {BOLD}{csv_image}{RESET}")
    print(f"    replicas         = {BOLD}{replicas}{RESET}")
    print(f"    dev volume       = {BOLD}{has_dev_volume}{RESET}")
    print()
    print(f"  {BOLD}Controller Deployment:{RESET}")
    print(f"    image = {BOLD}{deploy_image}{RESET}")
    print()

    # Model-serving-api
    server_image = cluster.get_deployment_image("model-serving-api", APPS_NS_L)
    print(f"  {BOLD}Model-serving-api Deployment:{RESET}")
    print(f"    image = {BOLD}{server_image}{RESET}")
    server_svc = cluster.resource_exists("service", "model-serving-api", "-n", APPS_NS_L)
    print(f"    service = {BOLD}{server_svc}{RESET}")
    print()

    # PVC
    print(f"  {BOLD}PVC{RESET} {PVC_NAME}:")
    print(f"    exists = {BOLD}{pvc_exists}{RESET}")
    print()

    # Runtime templates
    print(f"  {BOLD}Runtime templates{RESET} (all in {APPS_NS_L}):")
    result = oc("get", "templates", "-n", APPS_NS_L, "--no-headers", check=False)
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            print(f"    {parts[0]:<45s} {parts[1] if len(parts) > 1 else ''}")
    else:
        print("    (none)")
    print()

    # ConfigMap
    print(f"  {BOLD}ConfigMap{RESET} {CONFIGMAP_NAME}:")
    result = oc("get", "configmap", CONFIGMAP_NAME, "-n", APPS_NS_L, "-o", "json", check=False)
    if result.returncode == 0 and result.stdout.strip():
        cm = json.loads(result.stdout)
        data = cm.get("data", {})
        for k, v in sorted(data.items()):
            display = v if len(v) <= 75 else v[:72] + "..."
            print(f"    {k:45s} = {display}")
    else:
        print("    (configmap not found)")
    print()

    # Webhooks
    print(f"  {BOLD}Webhooks:{RESET}")
    mwh = cluster.resource_exists("mutatingwebhookconfiguration", "mutating.odh-model-controller.opendatahub.io")
    vwh = cluster.resource_exists("validatingwebhookconfiguration", "validating.odh-model-controller.opendatahub.io")
    wsvc = cluster.resource_exists("service", "odh-model-controller-webhook-service", "-n", APPS_NS_L)
    ok_or_miss("MutatingWebhookConfiguration", "true" if mwh else "not found")
    ok_or_miss("ValidatingWebhookConfiguration", "true" if vwh else "not found")
    ok_or_miss("Webhook Service", "true" if wsvc else "not found")
    print()

    # CRDs
    print(f"  {BOLD}CRDs:{RESET}")
    nim_crd = cluster.resource_exists("crd", "accounts.nim.opendatahub.io")
    ok_or_miss("accounts.nim.opendatahub.io", "true" if nim_crd else "not found")
    print()

    # RBAC
    print(f"  {BOLD}RBAC (key roles):{RESET}")
    ctrl_role = cluster.resource_exists("clusterrole", "odh-model-controller-role")
    srv_role = cluster.resource_exists("clusterrole", "model-serving-api")
    ok_or_miss("ClusterRole odh-model-controller-role", "true" if ctrl_role else "not found")
    ok_or_miss("ClusterRole model-serving-api", "true" if srv_role else "not found")
    print()

    # Pods
    _print_pods("Controller pod", APPS_NS_L, "app=odh-model-controller")
    _print_pods("Model-serving-api pod", APPS_NS_L, "app.kubernetes.io/name=model-serving-api")
    print()

    sf = state_file_for(cid)
    if sf.is_file():
        log.warning("Custom deploy state found for this cluster. Run 'undeploy' to restore.")
    else:
        log.info("No custom deploy state for this cluster — appears stock.")

    other_dirs = [
        d for d in sorted(STATE_DIR.iterdir())
        if d.is_dir() and d.name != cid and (d / "state.json").is_file()
    ] if STATE_DIR.is_dir() else []
    if other_dirs:
        print(f"\n  {BOLD}Active deployments on other clusters:{RESET}")
        for d in other_dirs:
            try:
                data = json.loads((d / "state.json").read_text())
                img = data.get("custom_controller_image", "?")
            except Exception:
                img = "?"
            print(f"    {d.name:<45s} {DIM}{img}{RESET}")


def _print_pods(label: str, ns: str, selector: str) -> None:
    print(f"  {BOLD}{label}:{RESET}")
    result = oc("get", "pods", "-n", ns, "-l", selector, "--no-headers", check=False)
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            name = parts[0] if parts else ""
            status = parts[2] if len(parts) > 2 else ""
            restarts = parts[3] if len(parts) > 3 else ""
            print(f"    {name:<50s} {status:<10s} {restarts}")
    else:
        print("    (not deployed)")


# ──────────────────────────── argparse + main ─────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deploy-custom-controller-on-operator.py",
        description=(
            "Deploy a custom odh-model-controller into a running RHOAI/ODH operator.\n\n"
            "Uses the PVC-based manifest override approach so the operator natively deploys\n"
            "your config/ manifests with full kustomize variable substitution. Works for any\n"
            "change — new runtimes, params updates, RBAC, webhooks, CRDs, server, etc.\n\n"
            "Ref: https://github.com/opendatahub-io/opendatahub-operator/blob/main/hack/component-dev/README.md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Build, push, and deploy in one step:
  %(prog)s build-deploy --tag vllm-omni-v1

  # Build both controller and server images:
  %(prog)s build-deploy --tag my-feature --with-server

  # Build from a repo at a different path:
  %(prog)s build-deploy --repo-root ~/src/odh-model-controller --tag my-fix

  # Deploy a pre-built image (no build step):
  %(prog)s deploy --controller-image quay.io/myuser/odh-model-controller:my-feature

  # Deploy without auto-revert (keep cluster state on failure for debugging):
  %(prog)s deploy --controller-image quay.io/myuser/odh-model-controller:debug --no-revert

  # Full cluster health check:
  %(prog)s status

  # Restore cluster to original state:
  %(prog)s undeploy
""",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose/debug logging (shows every oc command and its output)",
    )
    parser.add_argument(
        "--apps-ns", default=None,
        help="Namespace where controller runs (default: auto-detect)",
    )
    parser.add_argument(
        "--operator-ns", default=None,
        help="Namespace where the operator CSV lives (default: auto-detect)",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Non-interactive mode: skip all confirmation prompts",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # deploy
    p_deploy = sub.add_parser("deploy", help="Patch CSV, copy config/ manifests, restart operator")
    p_deploy.add_argument("--controller-image", help="Pre-built controller image to deploy")
    p_deploy.add_argument("--no-revert", action="store_true",
                          help="On failure, leave cluster as-is for debugging (default: auto-revert)")

    # build-deploy
    p_build = sub.add_parser("build-deploy", help="Build + push container image via Makefile, then deploy")
    p_build.add_argument("--repo-root", help="Path to odh-model-controller repo")
    p_build.add_argument("--tag", help="Short tag for quay.io/$USER/odh-model-controller:<tag>")
    p_build.add_argument("--image", help="Full image reference (mutually exclusive with --tag)")
    p_build.add_argument("--with-server", action="store_true",
                         help="Also build and push the model-serving-api image")
    p_build.add_argument("--no-revert", action="store_true",
                         help="On failure, leave cluster as-is for debugging (default: auto-revert)")

    # undeploy
    sub.add_parser("undeploy", help="Restore CSV, delete PVC, clean up extra templates")

    # status
    sub.add_parser("status", help="Full cluster health: deployments, templates, webhooks, CRDs, RBAC")

    return parser


def detect_namespaces(
    apps_ns_override: str | None, operator_ns_override: str | None,
) -> tuple[str, str]:
    """Auto-detect operator and apps namespaces, or use overrides."""
    if apps_ns_override and operator_ns_override:
        return apps_ns_override, operator_ns_override

    operator_ns = operator_ns_override or ""
    if not operator_ns:
        for candidate in ("redhat-ods-operator", "opendatahub", "openshift-operators"):
            result = oc(
                "get", "csv", "-n", candidate, "-o", "name", check=False,
            )
            for line in result.stdout.strip().splitlines():
                name = line.strip().lower()
                if "rhods-operator" in name or "rhoai-operator" in name or "opendatahub-operator" in name:
                    operator_ns = candidate
                    break
            if operator_ns:
                break
        if not operator_ns:
            operator_ns = OPERATOR_NS
            log.warning("Could not auto-detect operator namespace; defaulting to %s", operator_ns)

    apps_ns = apps_ns_override or ""
    if not apps_ns:
        if operator_ns == "redhat-ods-operator":
            apps_ns = "redhat-ods-applications"
        elif operator_ns == "opendatahub":
            apps_ns = "opendatahub"
        else:
            apps_ns = APPS_NS

    log.info("Using namespaces: operator=%s, apps=%s", operator_ns, apps_ns)
    return apps_ns, operator_ns


def main() -> None:
    global NON_INTERACTIVE
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)

    if args.yes:
        NON_INTERACTIVE = True

    if not args.command:
        parser.print_help()
        raise SystemExit(1)

    require_tools()

    apps_ns, operator_ns = detect_namespaces(
        getattr(args, "apps_ns", None),
        getattr(args, "operator_ns", None),
    )
    args.apps_ns = apps_ns
    args.operator_ns = operator_ns

    commands = {
        "deploy": cmd_deploy,
        "build-deploy": cmd_build_deploy,
        "undeploy": cmd_undeploy,
        "status": cmd_status,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
