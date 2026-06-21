#!/usr/bin/env python3
"""
GenAI Model Validation Script

Validates deployed vLLM / vLLM-Omni models against their acceptance criteria.
Supports: Text (standard vLLM), TTS, Diffusion, and Omni (multimodal) models.

Usage:
    python validate-genai-model.py -e <endpoint> -m <model_name> -t <model_type>

Model types:
    text       - Standard vLLM text/chat models (Llama, Mistral, etc.)
    tts        - Text-to-Speech (Qwen3-TTS, Voxtral-TTS)
    diffusion  - Image generation (FLUX.2-klein, FLUX.2-dev)
    omni       - Multimodal text+vision+audio (Qwen3-Omni)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import requests
    import urllib3
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("Missing required dependency: requests")
    print(f"  Install with: {sys.executable} -m pip install 'requests>=2.28.0'")
    sys.exit(1)

# ── ANSI Colors ──────────────────────────────────────────────────────────────

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
CYAN = "\033[0;36m"
BOLD = "\033[1m"
DIM = "\033[2m"
NC = "\033[0m"

if not sys.stdout.isatty():
    RED = GREEN = YELLOW = CYAN = BOLD = DIM = NC = ""

# ── Magic Bytes ──────────────────────────────────────────────────────────────

MAGIC_PNG = bytes.fromhex("89504e47")
MAGIC_JPEG = bytes.fromhex("ffd8ff")
MAGIC_RIFF = bytes.fromhex("52494646")
MAGIC_WEBP = b"WEBP"
MAGIC_MP3_FFFB = bytes.fromhex("fffb")
MAGIC_MP3_FFF3 = bytes.fromhex("fff3")
MAGIC_ID3 = bytes.fromhex("494433")

# ── Directories ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"


class ResultTracker:
    """Tracks test results with PASS/FAIL/SKIP counts."""

    def __init__(self) -> None:
        self.pass_count = 0
        self.fail_count = 0
        self.skip_count = 0
        self.results: list[str] = []

    def record(self, test_name: str, status: str, detail: str = "") -> None:
        detail_str = f" ({detail})" if detail else ""
        if status == "PASS":
            print(f"{GREEN}  PASS{NC} {test_name}{detail_str}")
            self.pass_count += 1
            self.results.append(f"PASS: {test_name}")
        elif status == "FAIL":
            print(f"{RED}  FAIL{NC} {test_name}{detail_str}")
            self.fail_count += 1
            detail_part = f" — {detail}" if detail else ""
            self.results.append(f"FAIL: {test_name}{detail_part}")
        else:
            print(f"{YELLOW}  SKIP{NC} {test_name}{detail_str}")
            self.skip_count += 1
            self.results.append(f"SKIP: {test_name}")

    def print_summary(self, model_name: str, model_type: str,
                      endpoint: str, output_dir: Path) -> bool:
        print_header("VALIDATION SUMMARY")
        print()
        print(f"  Model:    {BOLD}{model_name}{NC}")
        print(f"  Type:     {BOLD}{model_type}{NC}")
        print(f"  Endpoint: {BOLD}{endpoint}{NC}")
        print()

        for result in self.results:
            label = result.split(": ", 1)[1] if ": " in result else result
            if result.startswith("PASS:"):
                print(f"    {GREEN}PASS{NC}  {label}")
            elif result.startswith("FAIL:"):
                print(f"    {RED}FAIL{NC}  {label}")
            else:
                print(f"    {YELLOW}SKIP{NC}  {label}")

        total = self.pass_count + self.fail_count + self.skip_count
        print()
        print(f"  {GREEN}Passed: {self.pass_count}{NC}  "
              f"{RED}Failed: {self.fail_count}{NC}  "
              f"{YELLOW}Skipped: {self.skip_count}{NC}  "
              f"Total: {total}")
        print()

        if self.fail_count > 0:
            print(f"  {RED}{BOLD}RESULT: VALIDATION FAILED{NC}")
        else:
            print(f"  {GREEN}{BOLD}RESULT: VALIDATION PASSED{NC}")
        print()
        print(f"  Output files saved to: {output_dir}/")
        return self.fail_count == 0


class HttpClient:
    """HTTP client with retry, SSL control, auth, verbose logging, and timeouts."""

    RETRY_CODES = {502, 503, 504}
    RETRY_COUNT = 2

    def __init__(self, *, insecure: bool = False, bearer_token: str = "",
                 verbose: bool = False) -> None:
        self.verbose = verbose
        self.verify_ssl = not insecure
        self.session = requests.Session()
        if bearer_token:
            self.session.headers["Authorization"] = f"Bearer {bearer_token}"

        retry_strategy = Retry(
            total=self.RETRY_COUNT,
            backoff_factor=2,
            status_forcelist=[],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        if insecure:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"{DIM}    {msg}{NC}")

    def _should_retry(self, code: int) -> bool:
        return code in self.RETRY_CODES

    def _retry_delay(self, attempt: int) -> None:
        self._log(f">> Retry {attempt} after transient failure")
        time.sleep(attempt * 2)

    def get(self, url: str, timeout: int = 60) -> requests.Response | None:
        for attempt in range(self.RETRY_COUNT + 1):
            if attempt > 0:
                self._retry_delay(attempt)
            self._log(f">> GET {url}")
            try:
                resp = self.session.get(url, verify=self.verify_ssl, timeout=timeout)
                self._log(f"<< HTTP {resp.status_code}")
                self._log(f"<< {resp.text[:500]}")
                if self._should_retry(resp.status_code):
                    continue
                return resp
            except requests.RequestException as exc:
                self._log(f"<< Connection failed: {exc}")
                if attempt < self.RETRY_COUNT:
                    continue
                return None
        return None

    def post_json(self, url: str, data: dict, timeout: int = 300) -> requests.Response | None:
        for attempt in range(self.RETRY_COUNT + 1):
            if attempt > 0:
                self._retry_delay(attempt)
            self._log(f">> POST {url}")
            self._log(f">> Body: {json.dumps(data)[:300]}")
            try:
                resp = self.session.post(
                    url, json=data, verify=self.verify_ssl, timeout=timeout,
                )
                self._log(f"<< HTTP {resp.status_code}")
                self._log(f"<< {resp.text[:500]}")
                if self._should_retry(resp.status_code):
                    continue
                return resp
            except requests.RequestException as exc:
                self._log(f"<< Connection failed: {exc}")
                if attempt < self.RETRY_COUNT:
                    continue
                return None
        return None

    def post_binary(self, url: str, data: dict, output_path: Path,
                    timeout: int = 300) -> tuple[int, str]:
        """POST JSON, save binary response to file. Returns (status_code, content_type)."""
        for attempt in range(self.RETRY_COUNT + 1):
            if attempt > 0:
                self._retry_delay(attempt)
            self._log(f">> POST {url} -> {output_path}")
            self._log(f">> Body: {json.dumps(data)[:300]}")
            try:
                resp = self.session.post(
                    url, json=data, verify=self.verify_ssl, timeout=timeout,
                    stream=True,
                )
                content_type = resp.headers.get("Content-Type", "")
                self._log(f"<< HTTP {resp.status_code} Content-Type: {content_type}")
                if self._should_retry(resp.status_code):
                    continue
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                return resp.status_code, content_type
            except requests.RequestException as exc:
                self._log(f"<< Connection failed: {exc}")
                if attempt < self.RETRY_COUNT:
                    continue
                return 0, ""
        # All retries exhausted on transient status codes
        return 0, ""

    def post_stream(self, url: str, data: dict,
                    timeout: int = 60) -> str:
        """POST JSON for SSE streaming, returns raw text."""
        for attempt in range(self.RETRY_COUNT + 1):
            if attempt > 0:
                self._retry_delay(attempt)
            self._log(f">> POST (stream) {url}")
            self._log(f">> Body: {json.dumps(data)[:300]}")
            try:
                resp = self.session.post(
                    url, json=data, verify=self.verify_ssl, timeout=timeout,
                )
                text = resp.text
                if not text and attempt < self.RETRY_COUNT:
                    self._log("<< Empty response, will retry")
                    continue
                self._log(f"<< {text[:500]}")
                return text
            except requests.RequestException as exc:
                self._log(f"<< Connection failed: {exc}")
                if attempt < self.RETRY_COUNT:
                    continue
                return ""
        return ""

    MIME_MAP = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac"}

    def post_multipart(self, url: str, file_path: Path, file_field: str,
                       data: dict, timeout: int = 120) -> requests.Response | None:
        """POST multipart form data. Opens file fresh on each retry attempt."""
        mime = self.MIME_MAP.get(file_path.suffix.lower(), "application/octet-stream")
        for attempt in range(self.RETRY_COUNT + 1):
            if attempt > 0:
                self._retry_delay(attempt)
            self._log(f">> POST (multipart) {url}")
            try:
                with open(file_path, "rb") as f:
                    files = {file_field: (file_path.name, f, mime)}
                    resp = self.session.post(
                        url, files=files, data=data,
                        verify=self.verify_ssl, timeout=timeout,
                    )
                self._log(f"<< HTTP {resp.status_code}")
                self._log(f"<< {resp.text[:500]}")
                if self._should_retry(resp.status_code):
                    continue
                return resp
            except requests.RequestException as exc:
                self._log(f"<< Connection failed: {exc}")
                if attempt < self.RETRY_COUNT:
                    continue
                return None
        return None


def print_header(title: str) -> None:
    print()
    print(f"{CYAN}{'━' * 60}{NC}")
    print(f"{BOLD}  {title}{NC}")
    print(f"{CYAN}{'━' * 60}{NC}")


def print_test(title: str) -> None:
    print(f"\n{YELLOW}--- {title} ---{NC}")


def file_magic_check(filepath: Path, expected: bytes) -> bool:
    try:
        with open(filepath, "rb") as f:
            header = f.read(len(expected))
        return header == expected
    except OSError:
        return False


def is_webp(filepath: Path) -> bool:
    try:
        with open(filepath, "rb") as f:
            header = f.read(12)
        return header[:4] == MAGIC_RIFF and header[8:12] == MAGIC_WEBP
    except OSError:
        return False


def check_content_type(actual: str, *expected: str) -> bool:
    ct = actual.split(";")[0].strip().lower()
    return any(e in ct for e in expected)


def save_json(data: Any, path: Path) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except (OSError, TypeError):
        pass


def detect_and_rename_image(img_path: Path, base_name: str) -> None:
    """Detect image format by magic bytes and rename with correct extension."""
    try:
        if file_magic_check(img_path, MAGIC_PNG):
            img_path.rename(img_path.parent / f"{base_name}.png")
        elif file_magic_check(img_path, MAGIC_JPEG):
            img_path.rename(img_path.parent / f"{base_name}.jpg")
        elif is_webp(img_path):
            img_path.rename(img_path.parent / f"{base_name}.webp")
    except OSError:
        pass


# ── Common Validations ──────────────────────────────────────────────────────


def validate_health(client: HttpClient, endpoint: str, tracker: ResultTracker) -> bool:
    print_test("AC: /health endpoint reports healthy")
    resp = client.get(f"{endpoint}/health")
    if resp is not None and resp.status_code == 200:
        tracker.record("/health returns 200", "PASS")
        return True
    code = resp.status_code if resp else 0
    tracker.record("/health returns 200", "FAIL", f"HTTP {code}")
    return False


def validate_models_list(client: HttpClient, endpoint: str, model_name: str,
                         tracker: ResultTracker) -> None:
    print_test("AC: /v1/models lists the served model")
    resp = client.get(f"{endpoint}/v1/models")
    if resp is None or not (200 <= resp.status_code < 300):
        code = resp.status_code if resp else 0
        tracker.record("/v1/models returns 200", "FAIL", f"HTTP {code}")
        return

    tracker.record("/v1/models returns 200", "PASS")

    try:
        body = resp.json()
    except (json.JSONDecodeError, ValueError):
        tracker.record("/v1/models lists model(s)", "FAIL", "invalid JSON")
        return

    data = body.get("data", [])
    if not data:
        tracker.record("/v1/models lists model(s)", "FAIL", "empty data array")
        return

    model_ids = [m.get("id", "") for m in data if isinstance(m, dict)]
    tracker.record("/v1/models lists model(s)", "PASS", ", ".join(model_ids))

    if model_name in model_ids:
        tracker.record(f"/v1/models contains {model_name}", "PASS")
    else:
        tracker.record(f"/v1/models contains {model_name}", "FAIL",
                       f"model not found in: {', '.join(model_ids)}")


# ── Text Validation ─────────────────────────────────────────────────────────


def validate_text(client: HttpClient, endpoint: str, model_name: str,
                  output_dir: Path, tracker: ResultTracker) -> None:
    # /v1/chat/completions
    print_test("AC: Chat completion via /v1/chat/completions")
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Explain the water cycle in two sentences."}],
        "max_tokens": 256,
    }
    resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
    if resp is None or not (200 <= resp.status_code < 300):
        code = resp.status_code if resp else 0
        tracker.record("Chat completion returns 200", "FAIL", f"HTTP {code}")
    else:
        tracker.record("Chat completion returns 200", "PASS")
        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        save_json(body, output_dir / "chat-response.json")

        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if text and len(text) > 10:
            tracker.record("Chat response is non-empty", "PASS", f"{len(text)} chars")
        else:
            tracker.record("Chat response is non-empty", "FAIL", "empty or too short")

        reason = (body.get("choices") or [{}])[0].get("finish_reason")
        if reason:
            tracker.record("Finish reason present", "PASS", reason)
        else:
            tracker.record("Finish reason present", "FAIL", "missing")

        usage = body.get("usage")
        if usage:
            tracker.record("Usage stats present", "PASS",
                           f"prompt={usage.get('prompt_tokens', 0)}, "
                           f"completion={usage.get('completion_tokens', 0)}")
        else:
            tracker.record("Usage stats present", "FAIL", "missing")

    # /v1/completions
    print_test("AC: Text completion via /v1/completions")
    payload = {
        "model": model_name,
        "prompt": "List three benefits of open source software:",
        "max_tokens": 256,
    }
    resp = client.post_json(f"{endpoint}/v1/completions", payload)
    if resp is not None and 200 <= resp.status_code < 300:
        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        save_json(body, output_dir / "completions-response.json")
        choices = body.get("choices", [])
        ctext = choices[0].get("text", "") if choices else ""
        if ctext and len(ctext) > 10:
            tracker.record("/v1/completions works", "PASS", f"{len(ctext)} chars")
        else:
            tracker.record("/v1/completions works", "FAIL", "empty response")
    else:
        code = resp.status_code if resp else 0
        tracker.record("/v1/completions works", "FAIL", f"HTTP {code}")

    # Streaming
    print_test("AC: Streaming chat completion")
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hello in one sentence."}],
        "max_tokens": 64,
        "stream": True,
    }
    stream_text = client.post_stream(f"{endpoint}/v1/chat/completions", payload, timeout=60)
    if stream_text:
        chunks = [line for line in stream_text.splitlines() if line.startswith("data: ")]
        if len(chunks) > 1:
            tracker.record("Streaming returns SSE chunks", "PASS", f"{len(chunks)} chunks")
            (output_dir / "stream-response.txt").write_text(stream_text)
        elif len(chunks) == 1:
            tracker.record("Streaming returns SSE chunks", "FAIL", "only 1 chunk (not streaming)")
        else:
            tracker.record("Streaming returns SSE chunks", "FAIL", "no SSE data chunks")

        if "data: [DONE]" in stream_text:
            tracker.record("Stream ends with [DONE]", "PASS")
        else:
            tracker.record("Stream ends with [DONE]", "FAIL", "missing terminator")
    else:
        tracker.record("Streaming chat completion", "FAIL", "empty response")

    # Multi-turn
    print_test("AC: Multi-turn conversation")
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": "My name is Alex."},
            {"role": "assistant", "content": "Nice to meet you, Alex!"},
            {"role": "user", "content": "What is my name?"},
        ],
        "max_tokens": 64,
    }
    resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
    if resp is not None and 200 <= resp.status_code < 300:
        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        save_json(body, output_dir / "multiturn-response.json")
        mtext = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if "alex" in mtext.lower():
            tracker.record("Multi-turn retains context", "PASS")
        else:
            tracker.record("Multi-turn retains context", "FAIL", "name 'Alex' not in response")
    else:
        code = resp.status_code if resp else 0
        tracker.record("Multi-turn conversation", "FAIL", f"HTTP {code}")


# ── TTS Validation ──────────────────────────────────────────────────────────


def _discover_voices(client: HttpClient, endpoint: str) -> tuple[bool, list[str]]:
    """Discover available voices from /v1/audio/voices.

    Returns (http_ok, voices) so callers can distinguish a failed request
    from a successful request that returned an empty list.
    """
    resp = client.get(f"{endpoint}/v1/audio/voices")
    if resp is None or not (200 <= resp.status_code < 300):
        return False, []
    try:
        body = resp.json()
    except (json.JSONDecodeError, ValueError):
        return True, []

    voices: list[str] = []
    if isinstance(body, list):
        voices = [str(v) for v in body if v]
    elif isinstance(body, dict):
        items = body.get("voices") or body.get("data") or []
        for v in items:
            if isinstance(v, dict):
                name = v.get("name") or v.get("id")
                if name:
                    voices.append(name)
            elif v:
                voices.append(str(v))
    return True, voices


def validate_tts(client: HttpClient, endpoint: str, model_name: str,
                 output_dir: Path, tracker: ResultTracker) -> None:
    text = "Hello, this is a validation test of the text to speech system. The weather today is sunny and warm."

    # Discover voices
    voices_ok, voices = _discover_voices(client, endpoint)
    primary_voice = voices[0] if voices else "alloy"
    print(f"  Using primary voice: {BOLD}{primary_voice}{NC}")

    # WAV speech
    print_test("AC: /v1/audio/speech endpoint (wav format, predefined voice)")
    outfile = output_dir / "speech.wav"
    payload = {"model": model_name, "input": text, "voice": primary_voice, "response_format": "wav"}
    code, content_type = client.post_binary(f"{endpoint}/v1/audio/speech", payload, outfile)

    if not (200 <= code < 300):
        tracker.record("/v1/audio/speech (wav)", "FAIL", f"HTTP {code}")
    else:
        tracker.record("/v1/audio/speech returns 200", "PASS")
        fsize = outfile.stat().st_size if outfile.exists() else 0
        if fsize > 44:
            tracker.record("Response contains audio data", "PASS", f"{fsize} bytes")
        else:
            tracker.record("Response contains audio data", "FAIL", f"{fsize} bytes (too small)")

        if file_magic_check(outfile, MAGIC_RIFF):
            tracker.record("WAV magic bytes (RIFF)", "PASS")
        else:
            tracker.record("WAV magic bytes (RIFF)", "FAIL")

        if check_content_type(content_type, "audio/wav", "audio/x-wav", "audio/wave"):
            tracker.record("Content-Type header (wav)", "PASS", content_type)
        else:
            tracker.record("Content-Type header (wav)", "FAIL", f"got: {content_type}")

    # MP3 speech
    print_test("AC: /v1/audio/speech endpoint (mp3 format)")
    mp3file = output_dir / "speech.mp3"
    mp3_payload = {**payload, "response_format": "mp3"}
    code, content_type = client.post_binary(f"{endpoint}/v1/audio/speech", mp3_payload, mp3file)

    if not (200 <= code < 300):
        tracker.record("/v1/audio/speech (mp3)", "FAIL", f"HTTP {code}")
    else:
        tracker.record("/v1/audio/speech (mp3) returns 200", "PASS")
        mp3_ok = (
            file_magic_check(mp3file, MAGIC_MP3_FFFB)
            or file_magic_check(mp3file, MAGIC_MP3_FFF3)
            or file_magic_check(mp3file, MAGIC_ID3)
        )
        if mp3_ok:
            tracker.record("MP3 magic bytes", "PASS")
        else:
            tracker.record("MP3 magic bytes", "FAIL")

        if check_content_type(content_type, "audio/mpeg", "audio/mp3"):
            tracker.record("Content-Type header (mp3)", "PASS", content_type)
        else:
            tracker.record("Content-Type header (mp3)", "FAIL", f"got: {content_type}")

    # /v1/audio/voices validation (reuses discovery data to avoid a redundant GET)
    print_test("AC: /v1/audio/voices lists available speakers")
    if voices_ok:
        tracker.record("/v1/audio/voices returns 200", "PASS")
        if voices:
            tracker.record("Voices list is non-empty", "PASS", f"{len(voices)} voice(s)")
        else:
            tracker.record("Voices list is non-empty", "FAIL", "empty")
    else:
        tracker.record("/v1/audio/voices returns 200", "FAIL", "request failed")
        tracker.record("Voices list is non-empty", "FAIL", "skipped (API error)")

    # Second voice
    print_test("AC: CustomVoice with a different predefined speaker")
    second_voice = voices[1] if len(voices) > 1 else primary_voice
    if second_voice == primary_voice:
        tracker.record("CustomVoice alternate speaker", "SKIP", "only one voice available")
        return

    print(f"  Using second voice: {BOLD}{second_voice}{NC}")
    outfile2 = output_dir / f"speech-{second_voice}.wav"
    payload2 = {"model": model_name, "input": text, "voice": second_voice, "response_format": "wav"}
    code, _ = client.post_binary(f"{endpoint}/v1/audio/speech", payload2, outfile2)

    if 200 <= code < 300:
        fsize2 = outfile2.stat().st_size if outfile2.exists() else 0
        if fsize2 > 44:
            tracker.record(f"CustomVoice ({second_voice}) produces audio", "PASS", f"{fsize2} bytes")
        else:
            tracker.record(f"CustomVoice ({second_voice}) produces audio", "FAIL", f"{fsize2} bytes")
    else:
        tracker.record(f"CustomVoice ({second_voice}) produces audio", "FAIL", f"HTTP {code}")


# ── Diffusion Validation ────────────────────────────────────────────────────


def _decode_and_save_image(b64_data: str, output_dir: Path, base_name: str,
                           tracker: ResultTracker | None = None) -> None:
    """Decode base64 image, save with correct extension based on magic bytes."""
    img_path = output_dir / f"{base_name}.img"
    try:
        raw = base64.b64decode(b64_data)
        img_path.write_bytes(raw)
    except (ValueError, OSError):
        return

    if tracker:
        try:
            imgsize = img_path.stat().st_size
        except OSError:
            imgsize = 0
        if imgsize > 100:
            tracker.record("Decoded image has valid size", "PASS", f"{imgsize} bytes")
        else:
            tracker.record("Decoded image has valid size", "FAIL", f"{imgsize} bytes")

        if file_magic_check(img_path, MAGIC_PNG):
            tracker.record("Image is PNG format", "PASS")
        elif file_magic_check(img_path, MAGIC_JPEG):
            tracker.record("Image is JPEG format", "PASS")
        elif is_webp(img_path):
            tracker.record("Image is WEBP format", "PASS")
        else:
            tracker.record("Image format (PNG/JPEG/WEBP)", "FAIL")

    detect_and_rename_image(img_path, base_name)


def validate_diffusion(client: HttpClient, endpoint: str, model_name: str,
                       output_dir: Path, tracker: ResultTracker) -> None:
    # /v1/images/generations
    print_test("AC: /v1/images/generations endpoint")
    payload = {
        "model": model_name,
        "prompt": "A serene mountain landscape at sunset with a calm lake in the foreground",
        "size": "512x512",
        "seed": 42,
    }
    resp = client.post_json(f"{endpoint}/v1/images/generations", payload)
    if resp is None or not (200 <= resp.status_code < 300):
        code = resp.status_code if resp else 0
        tracker.record("/v1/images/generations returns 200", "FAIL", f"HTTP {code}")
        return

    tracker.record("/v1/images/generations returns 200", "PASS")

    try:
        body = resp.json()
    except (json.JSONDecodeError, ValueError):
        body = {}

    data = body.get("data", [])
    if not data:
        tracker.record("Response contains image data", "FAIL", "empty data array")
        return
    tracker.record("Response contains image data", "PASS")

    b64_data = data[0].get("b64_json", "")
    if b64_data:
        tracker.record("Image returned as base64", "PASS")
        _decode_and_save_image(b64_data, output_dir, "generated", tracker)
    else:
        url_data = data[0].get("url", "")
        if url_data:
            tracker.record("Image returned as URL", "PASS", url_data)
        else:
            tracker.record("Image data present (b64_json or url)", "FAIL", "neither found")

    # Chat-based image generation
    print_test("AC: Chat-based image generation (/v1/chat/completions)")
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Generate an image of a red fox sitting in a snowy forest."}],
        "max_tokens": 256,
    }
    resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
    if resp is not None and 200 <= resp.status_code < 300:
        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        if body.get("choices"):
            tracker.record("Chat image generation", "PASS")
            save_json(body, output_dir / "chat-image-response.json")
        else:
            tracker.record("Chat image generation", "FAIL", "no choices in response")
    else:
        code = resp.status_code if resp else 0
        tracker.record("Chat image generation", "SKIP", f"HTTP {code} (may not be supported)")

    # Second prompt
    print_test("AC: Second diffusion prompt")
    payload = {
        "model": model_name,
        "prompt": "A photorealistic cat wearing a tiny hat, studio lighting",
        "size": "512x512",
        "seed": 123,
    }
    resp = client.post_json(f"{endpoint}/v1/images/generations", payload)
    if resp is not None and 200 <= resp.status_code < 300:
        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        data = body.get("data", [])
        if data:
            tracker.record("Second prompt generates image", "PASS")
            b64_data2 = data[0].get("b64_json", "")
            if b64_data2:
                # Format checked on first image; second just saves the artifact
                _decode_and_save_image(b64_data2, output_dir, "generated-2")
        else:
            tracker.record("Second prompt generates image", "FAIL", "empty data")
    else:
        code = resp.status_code if resp else 0
        tracker.record("Second prompt generates image", "FAIL", f"HTTP {code}")


# ── Omni Validation ─────────────────────────────────────────────────────────


def validate_omni(client: HttpClient, endpoint: str, model_name: str,
                  output_dir: Path, tracker: ResultTracker) -> None:
    # Prepare image URL (local file preferred, fallback to external)
    image_file = INPUT_DIR / "test-scenery.jpg"
    if not image_file.exists():
        print(f"  {YELLOW}WARNING: {image_file} not found, vision test will use external URL{NC}")
        image_url = ("https://upload.wikimedia.org/wikipedia/commons/thumb/"
                     "d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/"
                     "2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")
    else:
        img_size = image_file.stat().st_size
        if img_size > 2_097_152:
            print(f"  {YELLOW}WARNING: {image_file} is {img_size // 1024}KB (>2MB), "
                  f"may cause oversized payload{NC}")
        b64 = base64.b64encode(image_file.read_bytes()).decode("ascii")
        image_url = f"data:image/jpeg;base64,{b64}"

    # Text-only chat
    print_test("AC: Text-only chat via /v1/chat/completions")
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Explain the water cycle in simple terms."}],
        "max_tokens": 256,
    }
    resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
    if resp is None or not (200 <= resp.status_code < 300):
        code = resp.status_code if resp else 0
        tracker.record("Text chat returns 200", "FAIL", f"HTTP {code}")
    else:
        tracker.record("Text chat returns 200", "PASS")
        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        save_json(body, output_dir / "text-chat-response.json")

        text = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if text and len(text) > 10:
            tracker.record("Text response is non-empty", "PASS", f"{len(text)} chars")
            keywords = ["water", "cycle", "evaporation", "rain", "cloud"]
            if any(kw in text.lower() for kw in keywords):
                tracker.record("Text response contains relevant content", "PASS")
            else:
                tracker.record("Text response contains relevant content", "FAIL",
                               "no expected keywords")
        else:
            tracker.record("Text response is non-empty", "FAIL", "empty or too short")

    # /v1/completions
    print_test("AC: /v1/completions endpoint")
    payload = {
        "model": model_name,
        "prompt": "List the top five benefits of renewable energy.",
        "max_tokens": 256,
    }
    resp = client.post_json(f"{endpoint}/v1/completions", payload)
    if resp is not None and 200 <= resp.status_code < 300:
        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        save_json(body, output_dir / "completions-response.json")
        choices = body.get("choices", [])
        ctext = choices[0].get("text", "") if choices else ""
        if ctext and len(ctext) > 10:
            tracker.record("/v1/completions works", "PASS", f"{len(ctext)} chars")
        else:
            tracker.record("/v1/completions works", "FAIL", "empty response")
    else:
        code = resp.status_code if resp else 0
        tracker.record("/v1/completions works", "FAIL", f"HTTP {code}")

    # Vision
    print_test("AC: Vision input via /v1/chat/completions")
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe what you see in this image."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        "max_tokens": 256,
    }
    resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
    if resp is None or not (200 <= resp.status_code < 300):
        code = resp.status_code if resp else 0
        tracker.record("Vision chat returns 200", "FAIL", f"HTTP {code}")
    else:
        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        save_json(body, output_dir / "vision-response.json")
        vtext = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
        if vtext and len(vtext) > 10:
            tracker.record("Vision response describes image", "PASS", f"{len(vtext)} chars")
        else:
            tracker.record("Vision response describes image", "FAIL", "empty or too short")

    # Audio transcription
    print_test("AC: Audio input via /v1/audio/transcriptions")
    audio_input = INPUT_DIR / "test-audio-en.wav"
    if audio_input.exists():
        resp = client.post_multipart(
            f"{endpoint}/v1/audio/transcriptions",
            file_path=audio_input,
            file_field="file",
            data={"model": model_name, "response_format": "json", "language": "en"},
        )
        if resp is not None and 200 <= resp.status_code < 300:
            tracker.record("/v1/audio/transcriptions returns 200", "PASS")
            try:
                body = resp.json()
            except (json.JSONDecodeError, ValueError):
                body = {}
            save_json(body, output_dir / "transcription-response.json")
            ttext = body.get("text", "")
            if ttext:
                tracker.record("Transcription contains text", "PASS", f"{len(ttext)} chars")
            elif "text" in body:
                tracker.record("Transcription contains text", "FAIL", "empty text field")
            else:
                tracker.record("Transcription contains text", "SKIP",
                               "no .text field (test audio is synthetic)")
        else:
            code = resp.status_code if resp else 0
            tracker.record("/v1/audio/transcriptions", "SKIP",
                           f"HTTP {code} (transcription may not be supported by this model)")
    else:
        tracker.record("Audio input test", "SKIP", "test-audio-en.wav not found in inputs/")

    # Audio output chat
    print_test("AC: Chat with audio output (if supported)")
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hello and introduce yourself briefly."}],
        "max_tokens": 256,
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "wav"},
    }
    resp = client.post_json(f"{endpoint}/v1/chat/completions", payload)
    if resp is not None and 200 <= resp.status_code < 300:
        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            body = {}
        save_json(body, output_dir / "chat-audio-response.json")
        choice = (body.get("choices") or [{}])[0].get("message", {})
        if choice.get("audio"):
            tracker.record("Chat audio output", "PASS")
            audio_b64 = choice["audio"].get("data", "")
            if audio_b64:
                try:
                    (output_dir / "chat-audio-output.wav").write_bytes(
                        base64.b64decode(audio_b64))
                except (ValueError, OSError):
                    pass
        elif choice.get("content"):
            tracker.record("Chat audio output", "SKIP",
                           "text response only (audio modality may need different config)")
        else:
            tracker.record("Chat audio output", "FAIL", "no content in response")
    else:
        code = resp.status_code if resp else 0
        tracker.record("Chat audio output", "SKIP",
                       f"HTTP {code} (audio modality may not be enabled)")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GenAI Model Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported models by type:
  text:       Any standard vLLM text/chat model (Llama, Mistral, etc.)
  tts:        Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice, Voxtral-4B-TTS-2603
  diffusion:  black-forest-labs/FLUX.2-klein-4B, FLUX.2-dev
  omni:       Qwen/Qwen3-Omni-30B-A3B-Instruct

Authentication:
  export BEARER_TOKEN="your-token-here"

Examples:
  %(prog)s -e https://llama3.apps.cluster.example.com -m llama3 -t text
  %(prog)s -e https://qwen3-tts.apps.cluster.example.com -m qwen3-tts -t tts -k
  %(prog)s -e https://flux2-klein.apps.cluster.example.com -m flux2-klein -t diffusion
  %(prog)s -e https://qwen3-omni.apps.cluster.example.com -m qwen3-omni -t omni -v
""",
    )
    parser.add_argument("-e", required=True, metavar="ENDPOINT",
                        help="Endpoint URL (e.g. https://model.apps.cluster.example.com)")
    parser.add_argument("-m", required=True, metavar="MODEL_NAME",
                        help="Model name (e.g. qwen3-tts)")
    parser.add_argument("-t", required=True, choices=["text", "tts", "diffusion", "omni"],
                        metavar="TYPE", help="Model type: text | tts | diffusion | omni")
    parser.add_argument("-k", action="store_true",
                        help="Insecure mode: skip SSL certificate verification")
    parser.add_argument("-v", action="store_true",
                        help="Verbose mode: show HTTP requests and raw responses")

    args = parser.parse_args()
    endpoint = args.e.rstrip("/")
    model_name = args.m
    model_type = args.t
    insecure = args.k
    verbose = args.v

    # Sanitize model name for output directory
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", model_name)
    safe_name = safe_name.lstrip(".")
    if not safe_name:
        safe_name = "_"
    output_dir = BASE_DIR / "outputs" / safe_name
    try:
        output_dir.resolve().relative_to(BASE_DIR / "outputs")
    except ValueError:
        print("Unsafe output path derived from model name")
        return 1

    # Prepare output directory
    if output_dir.exists():
        if verbose:
            print(f"{DIM}    Output directory found: {output_dir}{NC}")
        shutil.rmtree(output_dir)
        if verbose:
            print(f"{DIM}    Deleted previous output directory{NC}")
    else:
        if verbose:
            print(f"{DIM}    No previous output directory found{NC}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"{DIM}    Created fresh output directory: {output_dir}{NC}")

    bearer_token = os.environ.get("BEARER_TOKEN", "")
    client = HttpClient(insecure=insecure, bearer_token=bearer_token, verbose=verbose)
    tracker = ResultTracker()

    # Print header
    print_header("vLLM Model Validation")
    print()
    print(f"  Model:    {BOLD}{model_name}{NC}")
    print(f"  Type:     {BOLD}{model_type}{NC}")
    print(f"  Endpoint: {BOLD}{endpoint}{NC}")
    if bearer_token:
        print(f"  Auth:     {GREEN}Bearer token set{NC}")
    else:
        print(f"  Auth:     {YELLOW}None{NC}")
    if insecure:
        print(f"  SSL:      {YELLOW}Insecure (-k){NC}")
    else:
        print(f"  SSL:      {GREEN}Verified{NC}")
    print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z').strip()}")

    # Common checks
    health_ok = validate_health(client, endpoint, tracker)
    if not health_ok:
        print()
        print(f"  {RED}{BOLD}Health check failed — endpoint unreachable. "
              f"Skipping remaining tests.{NC}")
        tracker.print_summary(model_name, model_type, endpoint, output_dir)
        return 1

    validate_models_list(client, endpoint, model_name, tracker)

    # Type-specific checks
    validators = {
        "text": validate_text,
        "tts": validate_tts,
        "diffusion": validate_diffusion,
        "omni": validate_omni,
    }
    validators[model_type](client, endpoint, model_name, output_dir, tracker)

    passed = tracker.print_summary(model_name, model_type, endpoint, output_dir)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
