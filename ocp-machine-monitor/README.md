# OCP Machine Monitor

A production-grade CLI tool that monitors OpenShift Machine provisioning by instance type or MachineSet name, with optional Slack notifications on phase transitions.

Built for GPU machine pool provisioning workflows where operators need real-time visibility into machine lifecycle states across multiple availability zones.

## Features

- **Filter by instance type or MachineSet name** — monitor `p5.48xlarge`, `g5.12xlarge`, or specific MachineSet names
- **Slack notifications** — phase change alerts, stuck machine warnings, heartbeat pings, settlement notifications
- **Stuck detection** — configurable threshold with throttled re-alerts (one alert per threshold interval, not every poll)
- **Settlement detection** — auto-exits when all machines reach terminal states (Running, Failed, Mixed)
- **Min-running quorum** — exit early once N machines are Running without waiting for all
- **Timeout enforcement** — hard deadline with exit code 2
- **Single-shot mode** — `--once` for snapshot + exit (useful with `watch` or CI)
- **JSON output** — structured output to stdout (logs to stderr) for scripting
- **Status file** — atomic JSON writes for sidecar consumption (`watch cat status.json`)
- **Lock mode** — freeze the machine list after initial discovery (prevent mid-watch additions)
- **Dry-run** — full loop with message formatting but no Slack delivery
- **Colored terminal output** — ANSI colors with auto-detection and `--no-color`
- **Verbose mode** — machine conditions, events, addresses, last operation details
- **Graceful shutdown** — Ctrl+C prints final summary and sends Slack notification

## Prerequisites

- Python 3.10+
- `oc` CLI logged in to an OpenShift cluster
- (Optional) Slack incoming webhook URL

## Quick Start

```bash
# Monitor all p5.48xlarge machines (stdout only)
python3 ocp-machine-monitor.py -i p5.48xlarge

# Monitor multiple instance types with Slack
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T.../B.../xxx"
python3 ocp-machine-monitor.py -i p5.48xlarge -i g5.12xlarge

# Monitor a specific MachineSet
python3 ocp-machine-monitor.py -m my-gpu-machineset-us-east-1a

# Verbose, 15s poll, log to file
python3 ocp-machine-monitor.py -i p5.48xlarge -v -p 15 -l monitor.log

# Single snapshot as JSON
python3 ocp-machine-monitor.py -i p5.48xlarge --once --json

# Exit after 60 minutes or when 3 machines are running
python3 ocp-machine-monitor.py -i p5.48xlarge --timeout 60 --min-running 3

# Dry run (no Slack sent)
python3 ocp-machine-monitor.py -i p5.48xlarge --dry-run -s https://hooks.slack.com/...

# Heartbeat every 10 minutes, custom mention
python3 ocp-machine-monitor.py -i p5.48xlarge --heartbeat 10 --slack-mention '<!here>'

# Lock machine list, write status file for sidecar
python3 ocp-machine-monitor.py -i p5.48xlarge --lock-machines --status-file /tmp/gpu-status.json
```

## Usage

```
usage: ocp-machine-monitor.py [-h] [-i TYPE] [-m NAME] [-n NS]
                              [--lock-machines] [-p SECONDS] [--once]
                              [--timeout MINUTES] [--min-running N]
                              [--stuck-threshold MINUTES] [-s URL]
                              [--slack-mention MENTION] [--dry-run]
                              [--heartbeat MINUTES] [-v] [-l PATH] [--json]
                              [--no-color] [--status-file PATH]
```

### Filtering

| Flag | Description |
|------|-------------|
| `-i, --instance-type TYPE` | EC2/GCP instance type (repeatable) |
| `-m, --machineset NAME` | MachineSet name to monitor directly (repeatable) |
| `-n, --namespace NS` | Namespace for Machine API (default: `openshift-machine-api`) |
| `--lock-machines` | Disable dynamic re-discovery after initial snapshot |

### Polling

| Flag | Description |
|------|-------------|
| `-p, --poll SECONDS` | Poll interval (default: 30, min: 5) |
| `--once` | Single snapshot and exit |
| `--timeout MINUTES` | Exit with code 2 after N minutes (min: 1) |

### Settlement

| Flag | Description |
|------|-------------|
| `--min-running N` | Exit 0 once at least N machines are Running |
| `--stuck-threshold MINUTES` | Warn after N minutes in non-terminal phase (default: 15) |

### Slack

| Flag | Description |
|------|-------------|
| `-s, --slack URL` | Webhook URL (prefer `SLACK_WEBHOOK_URL` env var) |
| `--slack-mention MENTION` | Mention tag (default: `<!channel>`). Use `<!here>`, `<@USER_ID>`, or `""` |
| `--dry-run` | Full loop without sending Slack |
| `--heartbeat MINUTES` | Send "still watching" message every N minutes |

### Output

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Debug logging with conditions, events, addresses |
| `-l, --log-file PATH` | Append log output to file |
| `--json` | Emit JSON to stdout (logs go to stderr) |
| `--no-color` | Disable ANSI colors |
| `--status-file PATH` | Write JSON status file each poll (atomic writes) |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `SLACK_WEBHOOK_URL` | Slack webhook URL (preferred over `-s` for security) |
| `POLL_INTERVAL` | Default poll interval in seconds |

## Machine Phases

```
Provisioning → Provisioned → Running  (success)
                            → Failed   (unrecoverable, must delete)
Any phase    → Deleting     (deletion requested)
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All machines Running / min-running met / mixed-terminal / user stop |
| 1 | All machines Failed / `oc login` failure |
| 2 | Timeout reached (`--timeout`) |
| 3 | Still in progress (`--once` mode) |

## Security

- Webhook URLs passed via `-s` are visible in `ps aux` — use `SLACK_WEBHOOK_URL` env var instead
- Only `https://` webhook URLs are accepted
- OCP API data is escaped before injection into Slack `mrkdwn` payloads
- Slack messages are truncated at 38K chars (Slack limit is 40K)

## Architecture

- **Zero dependencies** — uses only Python stdlib (`argparse`, `json`, `subprocess`, `urllib`)
- **Single file** — no package installation required, just copy and run
- **Atomic status file writes** — write-to-tmp then `rename()` prevents partial reads
- **Exponential backoff with jitter** — on `oc` command failures
- **Signal-safe shutdown** — `SIGINT`/`SIGTERM` set a flag, main loop exits cleanly
- **Throttled stuck alerts** — one alert per `stuck-threshold` interval, not every poll
