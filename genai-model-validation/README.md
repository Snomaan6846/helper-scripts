# GenAI Model Validation

A comprehensive runtime certification tool for generative AI models deployed on RHOAI via vLLM and vLLM-Omni serving runtimes. Exercises OpenAI-compatible API endpoints with positive tests, negative tests, parameter variations, schema validation, and concurrency checks — producing a structured JSON report suitable for CI gating or Jira evidence.

## Features

- **Four model types** — text/chat (vLLM), TTS, diffusion, and omni/multimodal (vLLM-Omni)
- **Negative testing** — auth failure, malformed body, invalid model, empty input, boundary parameters
- **Schema validation** — verifies response JSON conforms to OpenAI API spec (field types, required fields)
- **Runtime metadata capture** — probes `/version`, `/health`, `/v1/models` to record vLLM version, GPU info, model dtype
- **Latency metrics** — per-request timing with p50/p95/max summary (informational, not a gate)
- **Concurrency smoke test** — 3 parallel requests to detect serialization bugs
- **Structured JSON report** — `results.json` always written for CI consumption
- **JUnit XML output** — `--output-format junit-xml` for CI pipeline integration
- **Test filtering** — `--tests` to run specific tests, `--skip-negative` for shared clusters
- **Dry-run mode** — `--dry-run` prints which tests would run without hitting endpoints
- **Warm-up request** — optional first request excluded from latency stats
- **SSL / Auth support** — `-k` for self-signed certs, `BEARER_TOKEN` env var for auth
- **Colored terminal output** — ANSI colors with auto-detection and `--no-color`
- **Verbose mode** — full HTTP request/response logging with `-v`
- **Keep outputs** — `--keep-outputs` for timestamped artifact directories across runs

## Prerequisites

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

For provisioning test input fixtures (images, audio), also requires system packages:

```bash
# Fedora/RHEL
sudo dnf install espeak-ng ffmpeg curl

# Then provision
./scripts/provision-inputs.sh
```

## Quick Start

```bash
# Standard vLLM text/chat model
python validate-genai-model.py -e https://llama3.apps.cluster.example.com -m llama3 -t text -k

# TTS model
python validate-genai-model.py -e https://qwen3-tts.apps.cluster.example.com -m qwen3-tts -t tts -k

# Diffusion model
python validate-genai-model.py -e https://flux2-klein.apps.cluster.example.com -m flux2-klein -t diffusion -k

# Omni model with verbose output
python validate-genai-model.py -e https://qwen3-omni.apps.cluster.example.com -m qwen3-omni -t omni -v -k

# With authentication
export BEARER_TOKEN="sha256~xxxxxx"
python validate-genai-model.py -e https://llama3.apps.cluster.example.com -m llama3 -t text -k

# Dry-run to see which tests would execute
python validate-genai-model.py -e https://any.example.com -m model -t text --dry-run

# Run only specific tests
python validate-genai-model.py -e https://llama3.example.com -m llama3 -t text -k --tests text_streaming,text_multiturn

# Skip negative tests on rate-limited clusters
python validate-genai-model.py -e https://llama3.example.com -m llama3 -t text -k --skip-negative

# Provision inputs and run omni validation
python validate-genai-model.py -e https://qwen3-omni.example.com -m qwen3-omni -t omni -k --provision-inputs
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-e` | Endpoint URL (required) | — |
| `-m` | Model name (required) | — |
| `-t` | Model type: `text`, `tts`, `diffusion`, `omni` (required) | — |
| `-k` | Skip SSL certificate verification | off |
| `-v` | Verbose mode (show HTTP requests/responses) | off |
| `--timeout N` | POST timeout in seconds; GET = max(30, N/5) | 300 |
| `--timeout-warn N` | Warn if any request exceeds N seconds | 30 |
| `--no-color` | Disable ANSI colors (supplements isatty detection) | auto |
| `--dry-run` | Print test list and exit without hitting endpoints | off |
| `--skip-negative` | Skip all negative/error tests | off |
| `--keep-outputs` | Use timestamped output subdirectory (don't delete previous) | off |
| `--provision-inputs` | Run `scripts/provision-inputs.sh` before validation | off |
| `--output-format` | Terminal output: `text`, `json`, or `junit-xml` | text |
| `--tests` | Comma-separated list of test names to run | all |
| `--exclude-tests` | Comma-separated list of test names to exclude | none |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All tests passed (SKIPs and WARNs are OK) |
| 1 | At least one test failed |
| 2 | Connectivity error (`/health` unreachable) |
| 3 | Config/input error (missing fixtures, invalid arguments, provision failure) |

## Project Structure

```
genai-model-validation/
├── README.md
├── requirements.txt               # Python dependencies (requests only)
├── validate-genai-model.py        # Validation script (single file, ~3000 lines)
├── scripts/
│   └── provision-inputs.sh        # Downloads/generates test fixtures (idempotent)
├── inputs/                        # Test input fixtures (downloaded, not committed)
│   ├── test-scenery.jpg           # Landscape for vision tests (1280x720)
│   ├── test-scenery-4k.jpg        # 4K image for oversize payload test (>2MB)
│   ├── test-image-2.jpg           # Second image for multi-image test
│   ├── test-audio-en.wav          # English speech for transcription (PCM WAV)
│   ├── test-audio-en.mp3          # Same content as MP3
│   └── test-audio-en.flac         # Same content as FLAC
└── outputs/                       # Per-model output directories (created at runtime)
    └── <model-name>/
        ├── results.json           # Structured test results (always written)
        ├── chat-response.json     # Text chat output
        ├── speech.wav             # TTS audio outputs
        ├── speech.mp3
        ├── speech-flac.flac
        └── generated.png          # Diffusion image outputs
```

## What Gets Validated

### Common (all model types)

- `/health` returns HTTP 200
- `/v1/models` returns 200 and lists the specified model
- Runtime metadata captured (version, GPU, dtype)
- Response schema validation (required fields, correct types)
- Auth failure test (invalid Bearer → 401/403; skipped if no token configured)
- Negative tests: invalid model name, malformed body, empty body, boundary `max_tokens`
- Concurrency smoke test (3 parallel requests)
- Warm-up request (excluded from latency stats)

### Text (standard vLLM)

- Chat completion with non-empty content, finish reason, usage stats
- Text completion via `/v1/completions`
- Streaming with SSE chunk validation (`delta.content` per chunk)
- Multi-turn conversation retains context
- System prompt adherence (JSON output validation)
- Token limit boundary (`max_tokens: 1` → `finish_reason: "length"`)
- Temperature variations (0 and 1.5)
- Stop sequences (`finish_reason: "stop"`)
- Logprobs (SKIP if disabled server-side)
- Long context (~4200 tokens)
- Function calling / tools (SKIP if unsupported)
- Invalid role and empty messages (negative tests)

### TTS (vLLM-Omni)

- WAV output with RIFF magic bytes and Content-Type validation
- MP3 output with magic bytes validation
- FLAC output with magic bytes validation
- Voice discovery via `/v1/audio/voices`
- Second voice (CustomVoice) produces valid audio
- WAV duration sanity check (PCM header parsing, >1s)
- Long text input (response >2x baseline bytes)
- Speed parameter variations (0.5, 2.0)
- Multi-language test
- Streaming audio (chunked transfer-encoding)
- Empty input and unsupported voice (negative tests)

### Diffusion (vLLM-Omni)

- Image generation with base64 response, PNG/JPEG/WEBP magic bytes
- Chat-based image generation
- Size matrix (256x256, 512x512, 1024x1024) with PNG dimension verification
- Seed reproducibility (same prompt+seed → identical output)
- Different prompts produce different images (caching bug detection)
- Batch generation (`n: 2`, iterates all items)
- Negative/guidance scale parameters
- URL response path validation (if supported)
- `num_inference_steps` variation
- Invalid size and empty prompt (negative tests)

### Omni (vLLM-Omni)

- Text-only chat with keyword relevance check
- Text completion via `/v1/completions`
- Vision input (image + text) with image description
- Multi-turn vision context retention
- Multiple images in one message
- Audio transcription (WAV, MP3, FLAC formats)
- Transcription content accuracy (keyword matching)
- Chat with audio output (WAV magic bytes check)
- Chat with MP3 audio output (magic bytes check)
- Modality combinations (`["text"]`, `["audio"]`, `["text", "audio"]`)
- Streaming + multimodal
- Large image input (>2MB oversize handling)
- Vision with URL vs base64
- Unsupported modality (negative test)

## Test Fixture Inputs

The `inputs/` directory contains test fixture files generated by `scripts/provision-inputs.sh`.

### Images

| File | Source | Resolution | Content |
|------|--------|------------|---------|
| `test-scenery.jpg` | Wikimedia CC0 | 1280x720 | Wisconsin boardwalk nature scene |
| `test-scenery-4k.jpg` | Wikimedia CC-BY-SA 3.0 | Original (>2 MB) | Fronalpstock mountain panorama |
| `test-image-2.jpg` | Wikimedia CC0 | 1280x720 | Cat photograph |

**Expected vision keywords**: `green`, `blue`, `sky`, `grass`, `path`, `tree`, `nature`, `water`, `lake`

### Audio

All audio files contain the spoken sentence: *"The quick brown fox jumps over the lazy dog"*

| File | Format | Generated with |
|------|--------|----------------|
| `test-audio-en.wav` | WAV (PCM) | `espeak-ng -v en -s 130` |
| `test-audio-en.mp3` | MP3, 128 kbps | ffmpeg transcode from WAV |
| `test-audio-en.flac` | FLAC (lossless) | ffmpeg transcode from WAV |

**Expected transcription keywords**: `quick`, `brown`, `fox`, `lazy`, `dog`

## JSON Report

Every run produces `outputs/<model>/results.json`:

```json
{
  "metadata": {
    "runtime_version": "0.8.5",
    "gpu": "NVIDIA H100",
    "model": "qwen3-omni",
    "model_type": "omni",
    "endpoint": "https://qwen3-omni.apps.cluster.example.com",
    "timestamp": "2026-06-24T23:30:00"
  },
  "latency": {
    "p50_ms": 245,
    "p95_ms": 1230,
    "max_ms": 3400,
    "count": 26
  },
  "tests": [
    {"name": "common_health", "status": "PASS", "latency_ms": 123, "detail": ""},
    {"name": "omni_transcription_accuracy", "status": "PASS", "latency_ms": 890, "detail": "4/5 keywords matched"}
  ],
  "summary": {"pass": 22, "fail": 0, "skip": 3, "warn": 1, "total": 26},
  "verdict": "PASS"
}
```

Verdict is `"PASS"` when `summary.fail == 0`, `"FAIL"` otherwise. SKIPs and WARNs do not affect the verdict.

## Test Naming Convention

Tests follow `{type}_{test}` in snake_case. Use `--dry-run` to see all available names:

```bash
$ python validate-genai-model.py -e http://x -m x -t tts --dry-run
Tests that would run for type 'tts':
  common_health
  common_models
  common_schema
  common_metadata
  common_negative_auth
  ...
  tts_wav
  tts_mp3
  tts_flac
  tts_voices
  tts_second_voice [dynamic]
  tts_long_text
  tts_wav_duration
  ...
```

## Adding New Models

To test a new model of an existing type, just pass the correct `-t` flag:

```bash
python validate-genai-model.py -e https://my-new-model.example.com -m my-model -t text -k
```

To add a new modality, create a `validate_<type>()` function in the script and add it to the `validators` dict and `TEST_REGISTRY` in `main()`.
