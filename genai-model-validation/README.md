# GenAI Model Validation

Manual validation toolkit for generative AI models deployed on RHOAI via vLLM and vLLM-Omni runtimes. Validates deployed models against their acceptance criteria by exercising OpenAI-compatible API endpoints and checking responses for correctness.

## Supported Model Types

| Type | Models | Endpoints Tested |
|------|--------|-----------------|
| **text** | Any standard vLLM text/chat model (Llama, Mistral, etc.) | `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions` (+ streaming, multi-turn) |
| **tts** | Qwen3-TTS-12Hz-1.7B-CustomVoice, Voxtral-4B-TTS-2603 | `/health`, `/v1/models`, `/v1/audio/speech`, `/v1/audio/voices` |
| **diffusion** | FLUX.2-klein-4B, FLUX.2-dev | `/health`, `/v1/models`, `/v1/images/generations`, `/v1/chat/completions` |
| **omni** | Qwen3-Omni-30B-A3B-Instruct | `/health`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `/v1/audio/transcriptions` |

## Prerequisites

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
genai-model-validation/
├── README.md
├── requirements.txt               # Python dependencies (requests)
├── validate-genai-model.py        # Validation script
├── inputs/                        # Test inputs for omni models
│   ├── test-scenery.jpg           # Image for vision tests
│   └── test-audio-en.wav          # Audio for transcription tests
└── outputs/                       # Generated per-model output directories (created at runtime)
    └── <model-name>/              # e.g. qwen3-tts/, flux2-klein/, llama3/
        ├── chat-response.json     # Text chat output
        ├── speech.wav             # TTS audio outputs
        ├── speech.mp3
        └── generated.png          # Diffusion image outputs
```

## Usage

```bash
python validate-genai-model.py -e <endpoint> -m <model_name> -t <model_type>
```

### Options

| Flag | Description | Example |
|------|-------------|---------|
| `-e` | Endpoint URL (required) | `https://llama3.apps.cluster.example.com` |
| `-m` | Model name (required) | `llama3` |
| `-t` | Model type (required) | `text`, `tts`, `diffusion`, or `omni` |
| `-k` | Insecure mode | Skip SSL certificate verification |
| `-v` | Verbose mode | Show HTTP requests and raw responses |
| `-h` | Show help | |

### Examples

```bash
# Standard vLLM text/chat model
python validate-genai-model.py \
  -e https://llama3.apps.cluster.example.com \
  -m llama3 \
  -t text

# TTS model with self-signed certs
python validate-genai-model.py \
  -e https://qwen3-tts.apps.cluster.example.com \
  -m qwen3-tts \
  -t tts \
  -k

# Diffusion model
python validate-genai-model.py \
  -e https://flux2-klein.apps.cluster.example.com \
  -m flux2-klein \
  -t diffusion \
  -k

# Omni model with verbose output
python validate-genai-model.py \
  -e https://qwen3-omni.apps.cluster.example.com \
  -m qwen3-omni \
  -t omni \
  -v -k

# With authentication
export BEARER_TOKEN="sha256~xxxxxx"
python validate-genai-model.py \
  -e https://llama3.apps.cluster.example.com \
  -m llama3 \
  -t text \
  -k
```

## What Gets Validated

### Common (all types)
- `/health` returns HTTP 200
- `/v1/models` returns 200 and lists at least one model
- `/v1/models` response contains the specified model name

### Text (standard vLLM)
- `/v1/chat/completions` returns 200 with non-empty content, finish reason, and usage stats
- `/v1/completions` returns 200 with generated text
- Streaming chat completion returns multiple SSE chunks and terminates with `[DONE]`
- Multi-turn conversation retains context across messages

### TTS
- `/v1/audio/speech` returns 200 with valid WAV audio (RIFF magic bytes)
- `/v1/audio/speech` returns 200 with valid MP3 audio
- `Content-Type` header matches the requested format (`audio/wav`, `audio/mpeg`)
- `/v1/audio/voices` returns a non-empty list of available speakers
- A second predefined voice produces valid audio output (CustomVoice)

### Diffusion
- `/v1/images/generations` returns 200 with base64-encoded image data
- Decoded image is valid PNG, JPEG, or WEBP format
- Chat-based image generation via `/v1/chat/completions` (if supported)
- Second prompt produces a different image

### Omni
- Text-only chat via `/v1/chat/completions` returns relevant content
- `/v1/completions` endpoint works for text generation
- Vision input (image + text) produces a description of the image
- Audio transcription via `/v1/audio/transcriptions` (if supported)
- Chat with audio output modality (if supported)

## Output

The script provides:

1. **Live terminal output** with color-coded PASS/FAIL/SKIP results
2. **Saved artifacts** in `outputs/<model-name>/` (audio files, images, JSON responses) for manual inspection or Jira evidence

## Adding New Models

To test a new model of an existing type, just pass the correct `-t` flag -- no script changes needed:

```bash
python validate-genai-model.py -e https://my-new-model.example.com -m my-model -t text
```

To add a new modality, create a `validate_<type>()` function in the script and add it to the `validators` dict in `main()`.
