# Helper Scripts

A collection of standalone utility scripts for OpenShift / Kubernetes operations and day-to-day platform engineering tasks.

## Scripts

| Script | Description |
|--------|-------------|
| [ocp-machine-monitor](ocp-machine-monitor/) | Monitor OpenShift Machine provisioning by instance type with Slack notifications |
| [genai-model-validation](genai-model-validation/) | Validate GenAI models (text, TTS, diffusion, omni) deployed on RHOAI via vLLM |

## Structure

Each script lives in its own directory with a dedicated README.

```
helper-scripts/
├── README.md
└── <script-name>/
    ├── <script-name>.py
    └── README.md
```

## Design Principles

- **Zero dependencies** — Python stdlib only, no `pip install` required
- **Portable** — copy a directory and run
- **Self-documented** — every script has `--help` and its own README

## Prerequisites

- Python 3.10+
- `oc` CLI (for OpenShift scripts)
