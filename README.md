# Helper Scripts

A collection of standalone utility scripts for OpenShift / Kubernetes operations and day-to-day platform engineering tasks.

## Scripts

| Script | Description |
|--------|-------------|
| [ocp-machine-monitor](ocp-machine-monitor/) | Monitor OpenShift Machine provisioning by instance type with Slack notifications |
| [genai-model-validation](genai-model-validation/) | Validate GenAI models (text, TTS, diffusion, omni) deployed on RHOAI via vLLM |
| [deploy-custom-controller-on-operator](deploy-custom-controller-on-operator/) | Deploy a custom odh-model-controller into a running RHOAI/ODH operator with auto-revert |

## Structure

Each script lives in its own directory with a dedicated README. Scripts that need third-party packages include their own `requirements.txt`.

```
helper-scripts/
├── README.md
└── <script-name>/
    ├── <script-name>.py
    ├── README.md
    └── requirements.txt  (optional)
```

## Design Principles

- **Portable** — copy a directory and run
- **Self-documented** — every script has `--help` and its own README
- **Minimal dependencies** — stdlib-only where possible; any extras listed in per-script `requirements.txt`

## Prerequisites

- Python 3.10+
- `oc` CLI (for OpenShift scripts)
