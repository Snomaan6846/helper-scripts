# Deploy Custom Controller on Operator

Deploy a custom `odh-model-controller` into a running RHOAI/ODH operator using the PVC-based manifest override approach. The operator natively deploys your `config/` manifests with full kustomize variable substitution — works for any change: new runtimes, params updates, RBAC, webhooks, CRDs, server, etc.

Based on the [upstream component-dev workflow](https://github.com/opendatahub-io/opendatahub-operator/blob/main/hack/component-dev/README.md).

## Features

- **Full lifecycle** — build, deploy, status, undeploy in one tool
- **Auto-revert on failure** — cluster is restored to stock if deploy fails mid-way (Ctrl+C safe)
- **Multi-cluster support** — per-cluster state and logging
- **ODH + RHOAI compatible** — auto-detects namespaces and operator type
- **CI-friendly** — `--yes` flag for non-interactive mode, exit code 2 on partial success
- **PVC-based manifest override** — operator natively renders kustomize with your manifests
- **CSV patching** — `RELATED_IMAGE` env var override so the operator's `ApplyParams` uses your image
- **Dynamic fsGroup** — queries namespace SCC supplemental-groups for restricted-v2 compatibility
- **Server image support** — `--with-server` builds and patches both controller and model-serving-api

## Prerequisites

- Python 3.10+
- `oc` CLI logged in to an OpenShift cluster with RHOAI or ODH installed
- (For `build-deploy`) Docker/Podman and access to a container registry

## Quick Start

```bash
# Build, push, and deploy in one step:
./deploy-custom-controller-on-operator.py build-deploy --tag my-feature

# Build both controller and server images:
./deploy-custom-controller-on-operator.py build-deploy --tag my-feature --with-server

# Build from a repo at a different path:
./deploy-custom-controller-on-operator.py build-deploy --repo-root ~/src/odh-model-controller --tag my-fix

# Deploy a pre-built image (no build step):
./deploy-custom-controller-on-operator.py deploy --controller-image quay.io/myuser/odh-model-controller:my-feature

# Deploy without auto-revert (keep cluster state on failure for debugging):
./deploy-custom-controller-on-operator.py deploy --controller-image quay.io/myuser/odh-model-controller:debug --no-revert

# Full cluster health check:
./deploy-custom-controller-on-operator.py status

# Restore cluster to original state:
./deploy-custom-controller-on-operator.py undeploy

# Non-interactive (CI):
./deploy-custom-controller-on-operator.py build-deploy --tag ci-test --yes
```

## Commands

### `build-deploy`

Builds and pushes the container image via the `odh-model-controller` Makefile, then deploys to the cluster.

| Flag | Description |
|------|-------------|
| `--repo-root` | Path to `odh-model-controller` repo (auto-detected if adjacent) |
| `--tag` | Short tag → `quay.io/$USER/odh-model-controller:<tag>` |
| `--image` | Full image reference (mutually exclusive with `--tag`) |
| `--with-server` | Also build and push `odh-model-serving-api`, patch it in `params.env` |
| `--no-revert` | On failure, leave cluster as-is for debugging |

### `deploy`

Deploys a pre-built image to the cluster (patches CSV, copies manifests, restarts operator).

| Flag | Description |
|------|-------------|
| `--controller-image` | Pre-built controller image to deploy |
| `--no-revert` | On failure, leave cluster as-is for debugging |

### `undeploy`

Restores the cluster to its original state: reverts CSV, deletes PVC, cleans up extra templates.

### `status`

Full cluster health snapshot: CSV state, deployment images, runtime templates, ConfigMap, webhooks, CRDs, RBAC, pod status.

## Global Options

| Flag | Description |
|------|-------------|
| `-v`, `--verbose` | Debug logging (shows every `oc` command and output) |
| `--apps-ns` | Override apps namespace (default: auto-detect) |
| `--operator-ns` | Override operator namespace (default: auto-detect) |
| `-y`, `--yes` | Non-interactive mode — skip all confirmation prompts |

## How It Works

1. **Discover CSV** — finds the RHOAI/ODH operator CSV and captures original state
2. **Create PVC** — `modelcontroller-manifests` in the operator namespace
3. **Patch CSV** — adds PVC volume mount, sets `RELATED_IMAGE_ODH_MODEL_CONTROLLER_IMAGE`, applies fsGroup
4. **Wait for operator** — PVC binds, operator pod restarts with the new volume
5. **Copy manifests** — `oc cp` of your local `config/` into the PVC
6. **Restart operator** — deletes pod (OLM-safe) so it re-renders kustomize with your manifests

On failure at any step, auto-revert calls `undeploy` to restore the cluster.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Fatal error (bad args, cluster not reachable, deploy failed and reverted) |
| 2 | Partial success — deploy applied but image propagation timed out (run `status` to check later) |

## State Management

Per-cluster state is stored in `.custom-deploy-state/<cluster-id>/`:
- `state.json` — captured original CSV values for undeploy
- `deploy.log` — full debug log of the deploy session
